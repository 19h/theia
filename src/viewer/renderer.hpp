#pragma once

#include "camera_controller.hpp"
#include "point_cloud_renderer.hpp"
#include "frustum_renderer.hpp"
#include "photo_compositor.hpp"

#include <psynth/types.hpp>
#include <memory>
#include <vector>
#include <string>

namespace psynth {
namespace sfm {
struct Reconstruction;
}  // namespace sfm

namespace viewer {

using Reconstruction = sfm::Reconstruction;

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool Initialize(int width, int height);
    void Resize(int width, int height);

    void LoadReconstruction(const Reconstruction& reconstruction,
                            const std::vector<ImageInfo>& images);
    void Clear();

    void Update(float delta_time, bool capture_input);
    void Render();

    // Input handling
    void OnKey(int key, int action);
    void OnMouseButton(int button, int action, double x, double y);
    void OnMouseMove(double x, double y);
    void OnScroll(double yoffset);

    // Camera control
    CameraController& GetCameraController() { return camera_controller_; }
    const CameraController& GetCameraController() const { return camera_controller_; }

    // Point cloud settings
    void SetPointSize(float size) { point_cloud_renderer_.SetPointSize(size); }
    float GetPointSize() const { return point_cloud_renderer_.GetPointSize(); }
    size_t GetPointCount() const { return point_cloud_renderer_.GetPointCount(); }

    // Frustum settings
    void SetFrustumScale(float scale) { frustum_renderer_.SetFrustumScale(scale); }
    float GetFrustumScale() const { return frustum_renderer_.GetFrustumScale(); }
    size_t GetCameraCount() const { return frustum_renderer_.GetFrustumCount(); }

    // Photo composite settings
    void SetBlendStrength(float strength) { photo_compositor_.SetBlendStrength(strength); }
    float GetBlendStrength() const { return photo_compositor_.GetBlendStrength(); }

    // Teleport to a camera
    void TeleportToCamera(int camera_index);

    // Enable/disable photo blending
    void SetPhotoBlendEnabled(bool enabled) { photo_blend_enabled_ = enabled; }
    bool IsPhotoBlendEnabled() const { return photo_blend_enabled_; }

    // Visibility toggles
    void SetPointCloudVisible(bool visible) { show_point_cloud_ = visible; }
    bool IsPointCloudVisible() const { return show_point_cloud_; }
    void SetFrustumsVisible(bool visible) { show_frustums_ = visible; }
    bool AreFrustumsVisible() const { return show_frustums_; }

private:
    void UpdateProximityPhoto();
    glm::vec3 ScreenToWorldRay(double screen_x, double screen_y);

    int width_ = 0;
    int height_ = 0;

    CameraController camera_controller_;
    PointCloudRenderer point_cloud_renderer_;
    FrustumRenderer frustum_renderer_;
    PhotoCompositor photo_compositor_;

    std::vector<ImageInfo> images_;

    // Mouse state
    bool right_mouse_down_ = false;
    double last_mouse_x_ = 0.0;
    double last_mouse_y_ = 0.0;
    bool first_mouse_ = true;

    // Render settings
    bool photo_blend_enabled_ = true;
    bool show_point_cloud_ = true;
    bool show_frustums_ = true;

    // Proximity threshold for auto-loading photos
    float proximity_threshold_ = 1.0f;
    int current_photo_index_ = -1;
};

}  // namespace viewer
}  // namespace psynth
