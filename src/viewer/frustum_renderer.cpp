#include "frustum_renderer.hpp"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <psynth/sfm/reconstruction.hpp>

#include <cmath>

namespace psynth {
namespace viewer {

namespace {

const char* kVertexShaderSource = R"(
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vColor;

void main() {
    gl_Position = uProjection * uView * vec4(aPos, 1.0);
    vColor = aColor;
}
)";

const char* kFragmentShaderSource = R"(
#version 410 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
)";

// Frustum vertices relative to camera center
// 5 vertices: center + 4 corners of the near plane
void GenerateFrustumLines(const FrustumData& frustum, float scale,
                          bool selected,
                          std::vector<float>& vertices) {
    glm::vec3 color = selected ? glm::vec3(1.0f, 1.0f, 0.0f) : glm::vec3(0.0f, 0.8f, 0.2f);

    float half_h = std::tan(frustum.fov_y * 0.5f) * scale;
    float half_w = half_h * frustum.aspect;

    // Camera-space corners of the "near plane"
    glm::vec3 tl(-half_w, half_h, -scale);
    glm::vec3 tr(half_w, half_h, -scale);
    glm::vec3 bl(-half_w, -half_h, -scale);
    glm::vec3 br(half_w, -half_h, -scale);

    // Transform to world space
    glm::mat3 R = frustum.rotation;
    glm::vec3 c = frustum.center;

    tl = c + R * tl;
    tr = c + R * tr;
    bl = c + R * bl;
    br = c + R * br;

    // Helper to add a line segment
    auto addLine = [&](const glm::vec3& a, const glm::vec3& b) {
        vertices.push_back(a.x); vertices.push_back(a.y); vertices.push_back(a.z);
        vertices.push_back(color.r); vertices.push_back(color.g); vertices.push_back(color.b);
        vertices.push_back(b.x); vertices.push_back(b.y); vertices.push_back(b.z);
        vertices.push_back(color.r); vertices.push_back(color.g); vertices.push_back(color.b);
    };

    // Lines from camera center to corners
    addLine(c, tl);
    addLine(c, tr);
    addLine(c, bl);
    addLine(c, br);

    // Rectangle at near plane
    addLine(tl, tr);
    addLine(tr, br);
    addLine(br, bl);
    addLine(bl, tl);
}

}  // namespace

FrustumRenderer::FrustumRenderer() = default;

FrustumRenderer::~FrustumRenderer() {
    Clear();
}

bool FrustumRenderer::Initialize() {
    if (!shader_.LoadFromSource(kVertexShaderSource, kFragmentShaderSource)) {
        return false;
    }

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    return true;
}

void FrustumRenderer::LoadFromReconstruction(const Reconstruction& reconstruction,
                                              const std::vector<ImageInfo>& images) {
    frustums_.clear();

    for (const auto& [image_id, camera] : reconstruction.cameras) {
        FrustumData fd;
        fd.image_id = image_id;

        // Find image path
        for (const auto& img : images) {
            if (img.id == image_id) {
                fd.image_path = img.path;
                break;
            }
        }

        // Camera center: C = -R^T * t
        Eigen::Vector3d center = CameraCenterWorld(camera.pose);
        fd.center = glm::vec3(
            static_cast<float>(center.x()),
            static_cast<float>(center.y()),
            static_cast<float>(center.z())
        );

        // Camera-to-world rotation: R^T
        Eigen::Matrix3d Rt = camera.pose.R.transpose();
        fd.rotation = glm::mat3(
            Rt(0,0), Rt(1,0), Rt(2,0),
            Rt(0,1), Rt(1,1), Rt(2,1),
            Rt(0,2), Rt(1,2), Rt(2,2)
        );

        // Compute FOV from intrinsics
        fd.fov_y = 2.0f * std::atan2(camera.intr.height_px * 0.5f, camera.intr.f_px);
        fd.aspect = static_cast<float>(camera.intr.width_px) / camera.intr.height_px;

        frustums_.push_back(fd);
    }

    BuildFrustumGeometry();
}

void FrustumRenderer::Clear() {
    frustums_.clear();

    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (vbo_) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    vertex_count_ = 0;
    selected_index_ = -1;
}

void FrustumRenderer::BuildFrustumGeometry() {
    std::vector<float> vertices;

    for (size_t i = 0; i < frustums_.size(); ++i) {
        bool selected = (static_cast<int>(i) == selected_index_);
        GenerateFrustumLines(frustums_[i], frustum_scale_, selected, vertices);
    }

    vertex_count_ = vertices.size() / 6;  // 6 floats per vertex (pos + color)

    if (vertex_count_ == 0) return;

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void FrustumRenderer::Render(const glm::mat4& view, const glm::mat4& projection) {
    if (vertex_count_ == 0 || !shader_.IsValid()) return;

    // Rebuild geometry if selection changed (to update colors)
    BuildFrustumGeometry();

    glEnable(GL_DEPTH_TEST);
    glLineWidth(2.0f);

    shader_.Use();
    shader_.SetMat4("uView", view);
    shader_.SetMat4("uProjection", projection);

    glBindVertexArray(vao_);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertex_count_));
    glBindVertexArray(0);
}

int FrustumRenderer::PickFrustum(const glm::vec3& ray_origin, const glm::vec3& ray_dir) const {
    int best_index = -1;
    float best_dist = std::numeric_limits<float>::max();

    // Simple sphere-based picking at camera centers
    const float pick_radius = frustum_scale_ * 0.3f;

    for (size_t i = 0; i < frustums_.size(); ++i) {
        glm::vec3 center = frustums_[i].center;
        glm::vec3 oc = ray_origin - center;

        float a = glm::dot(ray_dir, ray_dir);
        float b = 2.0f * glm::dot(oc, ray_dir);
        float c = glm::dot(oc, oc) - pick_radius * pick_radius;
        float discriminant = b * b - 4.0f * a * c;

        if (discriminant > 0) {
            float t = (-b - std::sqrt(discriminant)) / (2.0f * a);
            if (t > 0 && t < best_dist) {
                best_dist = t;
                best_index = static_cast<int>(i);
            }
        }
    }

    return best_index;
}

const FrustumData* FrustumRenderer::GetFrustumData(int index) const {
    if (index < 0 || index >= static_cast<int>(frustums_.size())) {
        return nullptr;
    }
    return &frustums_[index];
}

CameraState FrustumRenderer::GetCameraState(int index) const {
    CameraState state;
    if (index < 0 || index >= static_cast<int>(frustums_.size())) {
        return state;
    }

    const FrustumData& fd = frustums_[index];
    state.position = fd.center;
    state.orientation = glm::quat_cast(fd.rotation);
    state.fov = glm::degrees(fd.fov_y);

    return state;
}

}  // namespace viewer
}  // namespace psynth
