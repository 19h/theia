#include "renderer.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <psynth/sfm/reconstruction.hpp>

#include <cmath>

namespace psynth {
namespace viewer {

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

bool Renderer::Initialize(int width, int height) {
    width_ = width;
    height_ = height;

    if (!point_cloud_renderer_.Initialize()) {
        return false;
    }
    if (!frustum_renderer_.Initialize()) {
        return false;
    }
    if (!photo_compositor_.Initialize(width, height)) {
        return false;
    }

    return true;
}

void Renderer::Resize(int width, int height) {
    width_ = width;
    height_ = height;
    photo_compositor_.Resize(width, height);
}

void Renderer::LoadReconstruction(const Reconstruction& reconstruction,
                                   const std::vector<ImageInfo>& images) {
    images_ = images;
    point_cloud_renderer_.LoadFromReconstruction(reconstruction);
    frustum_renderer_.LoadFromReconstruction(reconstruction, images);

    // Position camera at first camera if available
    if (frustum_renderer_.GetFrustumCount() > 0) {
        CameraState initial = frustum_renderer_.GetCameraState(0);
        // Move back a bit from the first camera
        initial.position += initial.GetForward() * (-5.0f);
        camera_controller_.GetState() = initial;
    }
}

void Renderer::Clear() {
    point_cloud_renderer_.Clear();
    frustum_renderer_.Clear();
    photo_compositor_.ClearPhoto();
    images_.clear();
    current_photo_index_ = -1;
}

void Renderer::Update(float delta_time, bool capture_input) {
    camera_controller_.Update(delta_time, capture_input);

    if (photo_blend_enabled_) {
        UpdateProximityPhoto();
    }
}

void Renderer::Render() {
    const CameraState& cam = camera_controller_.GetState();
    float aspect = static_cast<float>(width_) / height_;
    glm::mat4 view = cam.GetViewMatrix();
    glm::mat4 projection = cam.GetProjectionMatrix(aspect);

    if (photo_blend_enabled_ && photo_compositor_.HasPhoto()) {
        // Render with photo compositing
        photo_compositor_.BeginDepthPass();

        if (show_point_cloud_) {
            point_cloud_renderer_.Render(view, projection);
        }
        if (show_frustums_) {
            frustum_renderer_.Render(view, projection);
        }

        photo_compositor_.EndDepthPass();

        glViewport(0, 0, width_, height_);
        photo_compositor_.CompositePhoto(photo_compositor_.GetBlendStrength());
    } else {
        // Direct rendering
        glViewport(0, 0, width_, height_);
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (show_point_cloud_) {
            point_cloud_renderer_.Render(view, projection);
        }
        if (show_frustums_) {
            frustum_renderer_.Render(view, projection);
        }
    }
}

void Renderer::OnKey(int key, int action) {
    camera_controller_.ProcessKeyboard(key, action);
}

void Renderer::OnMouseButton(int button, int action, double x, double y) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        right_mouse_down_ = (action == GLFW_PRESS);
        first_mouse_ = true;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        // Check for frustum picking
        glm::vec3 ray_dir = ScreenToWorldRay(x, y);
        glm::vec3 ray_origin = camera_controller_.GetState().position;

        int picked = frustum_renderer_.PickFrustum(ray_origin, ray_dir);
        if (picked >= 0) {
            TeleportToCamera(picked);
        }
    }
}

void Renderer::OnMouseMove(double x, double y) {
    if (right_mouse_down_) {
        if (first_mouse_) {
            last_mouse_x_ = x;
            last_mouse_y_ = y;
            first_mouse_ = false;
            return;
        }

        float xoffset = static_cast<float>(x - last_mouse_x_);
        float yoffset = static_cast<float>(last_mouse_y_ - y);  // Reversed: y goes down

        last_mouse_x_ = x;
        last_mouse_y_ = y;

        camera_controller_.ProcessMouseMovement(xoffset, yoffset);
    }
}

void Renderer::OnScroll(double yoffset) {
    camera_controller_.ProcessScroll(static_cast<float>(yoffset));
}

void Renderer::TeleportToCamera(int camera_index) {
    if (camera_index < 0 || camera_index >= static_cast<int>(frustum_renderer_.GetFrustumCount())) {
        return;
    }

    CameraState target = frustum_renderer_.GetCameraState(camera_index);
    camera_controller_.TeleportTo(target, 0.5f);
    frustum_renderer_.SetSelectedIndex(camera_index);

    // Load the photo for this camera
    const FrustumData* fd = frustum_renderer_.GetFrustumData(camera_index);
    if (fd && !fd->image_path.empty()) {
        photo_compositor_.LoadPhoto(fd->image_path);
        current_photo_index_ = camera_index;
    }
}

void Renderer::UpdateProximityPhoto() {
    const glm::vec3& pos = camera_controller_.GetState().position;

    int closest_index = -1;
    float closest_dist = proximity_threshold_;

    for (size_t i = 0; i < frustum_renderer_.GetFrustumCount(); ++i) {
        const FrustumData* fd = frustum_renderer_.GetFrustumData(static_cast<int>(i));
        if (!fd) continue;

        float dist = glm::length(fd->center - pos);
        if (dist < closest_dist) {
            closest_dist = dist;
            closest_index = static_cast<int>(i);
        }
    }

    if (closest_index != current_photo_index_) {
        if (closest_index >= 0) {
            const FrustumData* fd = frustum_renderer_.GetFrustumData(closest_index);
            if (fd && !fd->image_path.empty()) {
                photo_compositor_.LoadPhoto(fd->image_path);
                frustum_renderer_.SetSelectedIndex(closest_index);
            }
        } else {
            photo_compositor_.ClearPhoto();
            frustum_renderer_.SetSelectedIndex(-1);
        }
        current_photo_index_ = closest_index;
    }
}

glm::vec3 Renderer::ScreenToWorldRay(double screen_x, double screen_y) {
    // Convert screen coordinates to normalized device coordinates
    float x = (2.0f * static_cast<float>(screen_x)) / width_ - 1.0f;
    float y = 1.0f - (2.0f * static_cast<float>(screen_y)) / height_;

    const CameraState& cam = camera_controller_.GetState();
    float aspect = static_cast<float>(width_) / height_;

    glm::mat4 projection = cam.GetProjectionMatrix(aspect);
    glm::mat4 view = cam.GetViewMatrix();

    glm::mat4 inv_vp = glm::inverse(projection * view);

    glm::vec4 ray_clip(x, y, -1.0f, 1.0f);
    glm::vec4 ray_world = inv_vp * ray_clip;
    ray_world /= ray_world.w;

    glm::vec3 ray_dir = glm::normalize(glm::vec3(ray_world) - cam.position);

    return ray_dir;
}

}  // namespace viewer
}  // namespace psynth
