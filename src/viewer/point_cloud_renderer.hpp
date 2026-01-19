#pragma once

#include "shader.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace psynth {
namespace sfm {
struct Reconstruction;
}  // namespace sfm

namespace viewer {

using Reconstruction = sfm::Reconstruction;

class PointCloudRenderer {
public:
    PointCloudRenderer();
    ~PointCloudRenderer();

    PointCloudRenderer(const PointCloudRenderer&) = delete;
    PointCloudRenderer& operator=(const PointCloudRenderer&) = delete;

    bool Initialize();
    void LoadFromReconstruction(const Reconstruction& reconstruction);
    void Clear();

    void Render(const glm::mat4& view, const glm::mat4& projection);

    void SetPointSize(float size) { point_size_ = size; }
    float GetPointSize() const { return point_size_; }

    size_t GetPointCount() const { return point_count_; }

private:
    unsigned int vao_ = 0;
    unsigned int position_vbo_ = 0;
    unsigned int color_vbo_ = 0;
    size_t point_count_ = 0;

    Shader shader_;
    float point_size_ = 3.0f;
};

}  // namespace viewer
}  // namespace psynth
