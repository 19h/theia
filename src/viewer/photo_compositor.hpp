#pragma once

#include "shader.hpp"
#include <glm/glm.hpp>
#include <string>

namespace psynth {
namespace viewer {

class PhotoCompositor {
public:
    PhotoCompositor();
    ~PhotoCompositor();

    PhotoCompositor(const PhotoCompositor&) = delete;
    PhotoCompositor& operator=(const PhotoCompositor&) = delete;

    bool Initialize(int width, int height);
    void Resize(int width, int height);

    // Load a photo texture from file
    bool LoadPhoto(const std::string& path);
    void ClearPhoto();

    // Render pipeline:
    // 1. Call BeginDepthPass() before rendering point cloud
    // 2. Render point cloud normally
    // 3. Call EndDepthPass()
    // 4. Call CompositePhoto() to blend photo over point cloud
    void BeginDepthPass();
    void EndDepthPass();
    void CompositePhoto(float blend_strength);

    bool HasPhoto() const { return photo_texture_ != 0; }

    void SetBlendStrength(float strength) { blend_strength_ = strength; }
    float GetBlendStrength() const { return blend_strength_; }

    // Depth FBO for external access
    unsigned int GetDepthTexture() const { return depth_texture_; }
    unsigned int GetColorTexture() const { return color_texture_; }

private:
    void CreateFramebuffer(int width, int height);
    void DeleteFramebuffer();
    void CreateFullscreenQuad();

    int width_ = 0;
    int height_ = 0;

    // Offscreen FBO for depth capture
    unsigned int fbo_ = 0;
    unsigned int color_texture_ = 0;
    unsigned int depth_texture_ = 0;

    // Photo texture
    unsigned int photo_texture_ = 0;
    int photo_width_ = 0;
    int photo_height_ = 0;

    // Fullscreen quad
    unsigned int quad_vao_ = 0;
    unsigned int quad_vbo_ = 0;

    Shader composite_shader_;
    float blend_strength_ = 0.7f;
};

}  // namespace viewer
}  // namespace psynth
