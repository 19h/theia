#pragma once

#include "shader.hpp"
#include "camera_controller.hpp"
#include <psynth/types.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <string>

namespace psynth {
namespace sfm {
struct Reconstruction;
}  // namespace sfm

namespace viewer {

using Reconstruction = sfm::Reconstruction;

struct FrustumData {
    glm::vec3 center;
    glm::mat3 rotation;  // Camera-to-world rotation
    float fov_y;
    float aspect;
    ImageId image_id;
    std::string image_path;
};

class FrustumRenderer {
public:
    FrustumRenderer();
    ~FrustumRenderer();

    FrustumRenderer(const FrustumRenderer&) = delete;
    FrustumRenderer& operator=(const FrustumRenderer&) = delete;

    bool Initialize();
    void LoadFromReconstruction(const Reconstruction& reconstruction,
                                 const std::vector<ImageInfo>& images);
    void Clear();

    void Render(const glm::mat4& view, const glm::mat4& projection);

    // Picking
    int PickFrustum(const glm::vec3& ray_origin, const glm::vec3& ray_dir) const;
    const FrustumData* GetFrustumData(int index) const;

    // Selection
    void SetSelectedIndex(int index) { selected_index_ = index; }
    int GetSelectedIndex() const { return selected_index_; }

    // Convert frustum to CameraState for teleportation
    CameraState GetCameraState(int index) const;

    // Settings
    void SetFrustumScale(float scale) { frustum_scale_ = scale; }
    float GetFrustumScale() const { return frustum_scale_; }

    size_t GetFrustumCount() const { return frustums_.size(); }

private:
    void BuildFrustumGeometry();

    std::vector<FrustumData> frustums_;

    unsigned int vao_ = 0;
    unsigned int vbo_ = 0;
    size_t vertex_count_ = 0;

    Shader shader_;
    float frustum_scale_ = 0.5f;
    int selected_index_ = -1;
};

}  // namespace viewer
}  // namespace psynth
