#include "photo_compositor.hpp"

#include <glad/glad.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>

namespace psynth {
namespace viewer {

namespace {

const char* kCompositeVertexShader = R"(
#version 410 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
)";

const char* kCompositeFragmentShader = R"(
#version 410 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uSceneColor;
uniform sampler2D uSceneDepth;
uniform sampler2D uPhoto;
uniform float uBlendStrength;

void main() {
    vec4 sceneColor = texture(uSceneColor, vTexCoord);
    float depth = texture(uSceneDepth, vTexCoord).r;
    vec4 photoColor = texture(uPhoto, vTexCoord);

    // Blend photo with scene based on blend strength
    // When depth is far (close to 1.0), show more photo
    // When depth is near (close to 0.0), show more scene (point cloud)
    float depthFactor = smoothstep(0.0, 0.99, depth);
    float blend = uBlendStrength * depthFactor;

    FragColor = mix(sceneColor, photoColor, blend);
}
)";

}  // namespace

PhotoCompositor::PhotoCompositor() = default;

PhotoCompositor::~PhotoCompositor() {
    ClearPhoto();
    DeleteFramebuffer();

    if (quad_vao_) {
        glDeleteVertexArrays(1, &quad_vao_);
    }
    if (quad_vbo_) {
        glDeleteBuffers(1, &quad_vbo_);
    }
}

bool PhotoCompositor::Initialize(int width, int height) {
    if (!composite_shader_.LoadFromSource(kCompositeVertexShader, kCompositeFragmentShader)) {
        return false;
    }

    CreateFullscreenQuad();
    CreateFramebuffer(width, height);

    return true;
}

void PhotoCompositor::Resize(int width, int height) {
    if (width == width_ && height == height_) return;
    DeleteFramebuffer();
    CreateFramebuffer(width, height);
}

bool PhotoCompositor::LoadPhoto(const std::string& path) {
    ClearPhoto();

    int channels;
    unsigned char* data = stbi_load(path.c_str(), &photo_width_, &photo_height_, &channels, 4);
    if (!data) {
        std::cerr << "Failed to load photo: " << path << std::endl;
        return false;
    }

    glGenTextures(1, &photo_texture_);
    glBindTexture(GL_TEXTURE_2D, photo_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, photo_width_, photo_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    stbi_image_free(data);
    return true;
}

void PhotoCompositor::ClearPhoto() {
    if (photo_texture_) {
        glDeleteTextures(1, &photo_texture_);
        photo_texture_ = 0;
    }
    photo_width_ = 0;
    photo_height_ = 0;
}

void PhotoCompositor::BeginDepthPass() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, width_, height_);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void PhotoCompositor::EndDepthPass() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PhotoCompositor::CompositePhoto(float blend_strength) {
    if (!photo_texture_) {
        // No photo - just render the scene texture
        blend_strength = 0.0f;
    }

    glDisable(GL_DEPTH_TEST);

    composite_shader_.Use();
    composite_shader_.SetInt("uSceneColor", 0);
    composite_shader_.SetInt("uSceneDepth", 1);
    composite_shader_.SetInt("uPhoto", 2);
    composite_shader_.SetFloat("uBlendStrength", blend_strength);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, color_texture_);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depth_texture_);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, photo_texture_ ? photo_texture_ : color_texture_);

    glBindVertexArray(quad_vao_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
}

void PhotoCompositor::CreateFramebuffer(int width, int height) {
    width_ = width;
    height_ = height;

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

    // Color texture
    glGenTextures(1, &color_texture_);
    glBindTexture(GL_TEXTURE_2D, color_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture_, 0);

    // Depth texture
    glGenTextures(1, &depth_texture_);
    glBindTexture(GL_TEXTURE_2D, depth_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture_, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PhotoCompositor::DeleteFramebuffer() {
    if (fbo_) {
        glDeleteFramebuffers(1, &fbo_);
        fbo_ = 0;
    }
    if (color_texture_) {
        glDeleteTextures(1, &color_texture_);
        color_texture_ = 0;
    }
    if (depth_texture_) {
        glDeleteTextures(1, &depth_texture_);
        depth_texture_ = 0;
    }
}

void PhotoCompositor::CreateFullscreenQuad() {
    float quad_vertices[] = {
        // positions   // texcoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
    };

    glGenVertexArrays(1, &quad_vao_);
    glGenBuffers(1, &quad_vbo_);

    glBindVertexArray(quad_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

}  // namespace viewer
}  // namespace psynth
