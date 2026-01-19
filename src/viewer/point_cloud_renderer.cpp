#include "point_cloud_renderer.hpp"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <psynth/sfm/reconstruction.hpp>

namespace psynth {
namespace viewer {

namespace {

const char* kVertexShaderSource = R"(
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uPointSize;

out vec3 vColor;

void main() {
    gl_Position = uProjection * uView * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
}
)";

const char* kFragmentShaderSource = R"(
#version 410 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    // Circular points
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) {
        discard;
    }
    FragColor = vec4(vColor, 1.0);
}
)";

}  // namespace

PointCloudRenderer::PointCloudRenderer() = default;

PointCloudRenderer::~PointCloudRenderer() {
    Clear();
}

bool PointCloudRenderer::Initialize() {
    if (!shader_.LoadFromSource(kVertexShaderSource, kFragmentShaderSource)) {
        return false;
    }

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &position_vbo_);
    glGenBuffers(1, &color_vbo_);

    return true;
}

void PointCloudRenderer::LoadFromReconstruction(const Reconstruction& reconstruction) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> colors;

    for (const auto& track : reconstruction.tracks.all()) {
        if (!track.triangulated) continue;

        // Convert Eigen to glm
        positions.push_back(glm::vec3(
            static_cast<float>(track.xyz.x()),
            static_cast<float>(track.xyz.y()),
            static_cast<float>(track.xyz.z())
        ));

        // Convert BGR to RGB, normalize to [0,1]
        colors.push_back(glm::vec3(
            track.color_bgr[2] / 255.0f,  // R (was B)
            track.color_bgr[1] / 255.0f,  // G
            track.color_bgr[0] / 255.0f   // B (was R)
        ));
    }

    point_count_ = positions.size();
    if (point_count_ == 0) return;

    glBindVertexArray(vao_);

    // Upload positions
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo_);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), positions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
    glEnableVertexAttribArray(0);

    // Upload colors
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), colors.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void PointCloudRenderer::Clear() {
    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (position_vbo_) {
        glDeleteBuffers(1, &position_vbo_);
        position_vbo_ = 0;
    }
    if (color_vbo_) {
        glDeleteBuffers(1, &color_vbo_);
        color_vbo_ = 0;
    }
    point_count_ = 0;
}

void PointCloudRenderer::Render(const glm::mat4& view, const glm::mat4& projection) {
    if (point_count_ == 0 || !shader_.IsValid()) return;

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);

    shader_.Use();
    shader_.SetMat4("uView", view);
    shader_.SetMat4("uProjection", projection);
    shader_.SetFloat("uPointSize", point_size_);

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count_));
    glBindVertexArray(0);
}

}  // namespace viewer
}  // namespace psynth
