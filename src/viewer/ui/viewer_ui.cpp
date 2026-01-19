#include "viewer_ui.hpp"
#include "../renderer.hpp"

#include <imgui.h>

#include <algorithm>

namespace psynth {
namespace viewer {

namespace {
bool EndsWith(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
}  // namespace

ViewerUI::ViewerUI() = default;
ViewerUI::~ViewerUI() = default;

void ViewerUI::Render() {
    wants_capture_input_ = ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard;

    RenderMenuBar();
    RenderSidePanel();
    RenderStatusBar();
}

void ViewerUI::AddDroppedFiles(const std::vector<std::string>& paths) {
    for (const auto& path : paths) {
        // Check if it's an image file
        std::string lower = path;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (EndsWith(lower, ".jpg") || EndsWith(lower, ".jpeg") ||
            EndsWith(lower, ".png") || EndsWith(lower, ".bmp") ||
            EndsWith(lower, ".tiff") || EndsWith(lower, ".tif")) {
            dropped_files_.push_back(path);
        }
        // Check if it's a reconstruction file
        else if (EndsWith(lower, ".json") || EndsWith(lower, ".bin")) {
            if (load_callback_) {
                load_callback_(path);
            }
        }
    }
}

void ViewerUI::RenderMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Reconstruction...", "Ctrl+O")) {
                // TODO: File dialog
            }
            if (ImGui::MenuItem("Export PLY...", "Ctrl+E")) {
                if (export_callback_) {
                    export_callback_("output.ply");
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Quit", "Ctrl+Q")) {
                // Will be handled by window
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (renderer_) {
                bool show_points = renderer_->IsPointCloudVisible();
                if (ImGui::MenuItem("Point Cloud", nullptr, &show_points)) {
                    renderer_->SetPointCloudVisible(show_points);
                }

                bool show_frustums = renderer_->AreFrustumsVisible();
                if (ImGui::MenuItem("Camera Frustums", nullptr, &show_frustums)) {
                    renderer_->SetFrustumsVisible(show_frustums);
                }

                bool photo_blend = renderer_->IsPhotoBlendEnabled();
                if (ImGui::MenuItem("Photo Blending", nullptr, &photo_blend)) {
                    renderer_->SetPhotoBlendEnabled(photo_blend);
                }
            }
            ImGui::Separator();
            ImGui::MenuItem("ImGui Demo", nullptr, &show_demo_window_);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }

    if (show_demo_window_) {
        ImGui::ShowDemoWindow(&show_demo_window_);
    }
}

void ViewerUI::RenderSidePanel() {
    ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 320, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, io.DisplaySize.y - 60), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse)) {
        // Drop zone
        RenderDropZone();

        ImGui::Separator();

        // Render settings
        if (ImGui::CollapsingHeader("Render Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (renderer_) {
                float point_size = renderer_->GetPointSize();
                if (ImGui::SliderFloat("Point Size", &point_size, 1.0f, 10.0f)) {
                    renderer_->SetPointSize(point_size);
                }

                float frustum_scale = renderer_->GetFrustumScale();
                if (ImGui::SliderFloat("Frustum Scale", &frustum_scale, 0.1f, 2.0f)) {
                    renderer_->SetFrustumScale(frustum_scale);
                }

                float blend_strength = renderer_->GetBlendStrength();
                if (ImGui::SliderFloat("Photo Blend", &blend_strength, 0.0f, 1.0f)) {
                    renderer_->SetBlendStrength(blend_strength);
                }
            }
        }

        // Camera info
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (renderer_) {
                const CameraState& cam = renderer_->GetCameraController().GetState();
                ImGui::Text("Position: (%.2f, %.2f, %.2f)", cam.position.x, cam.position.y, cam.position.z);
                ImGui::Text("FOV: %.1f deg", cam.fov);
                ImGui::Text("Speed: %.1f", renderer_->GetCameraController().movement_speed);

                if (renderer_->GetCameraController().IsTransitioning()) {
                    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Transitioning...");
                }
            }
        }

        // Stats
        if (ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (renderer_) {
                ImGui::Text("Points: %zu", renderer_->GetPointCount());
                ImGui::Text("Cameras: %zu", renderer_->GetCameraCount());
            }
            ImGui::Text("FPS: %.1f", io.Framerate);
        }

        // Camera list
        if (renderer_ && renderer_->GetCameraCount() > 0) {
            if (ImGui::CollapsingHeader("Cameras")) {
                for (size_t i = 0; i < renderer_->GetCameraCount(); ++i) {
                    char label[64];
                    snprintf(label, sizeof(label), "Camera %zu", i);

                    if (ImGui::Selectable(label)) {
                        renderer_->TeleportToCamera(static_cast<int>(i));
                    }
                }
            }
        }

        // Controls help
        if (ImGui::CollapsingHeader("Controls")) {
            ImGui::BulletText("WASD: Move");
            ImGui::BulletText("Space/Shift: Up/Down");
            ImGui::BulletText("Right-click + drag: Look");
            ImGui::BulletText("Scroll: Adjust speed");
            ImGui::BulletText("Left-click frustum: Teleport");
        }
    }
    ImGui::End();
}

void ViewerUI::RenderDropZone() {
    ImGui::Text("Drag images here to reconstruct:");

    ImVec2 size(ImGui::GetContentRegionAvail().x, 80);
    ImGui::BeginChild("DropZone", size, true);

    if (dropped_files_.empty()) {
        ImGui::TextDisabled("Drop image files...");
    } else {
        ImGui::Text("%zu images queued", dropped_files_.size());

        if (!reconstruction_running_) {
            if (ImGui::Button("Start Reconstruction", ImVec2(-1, 0))) {
                if (reconstruct_callback_) {
                    reconstruct_callback_(dropped_files_);
                    dropped_files_.clear();
                }
            }
        } else {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Reconstructing...");
        }

        if (ImGui::Button("Clear", ImVec2(-1, 0))) {
            dropped_files_.clear();
        }
    }

    ImGui::EndChild();
}

void ViewerUI::RenderStatusBar() {
    ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos(ImVec2(0, io.DisplaySize.y - 25));
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 25));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("StatusBar", nullptr, flags)) {
        ImGui::Text("%s", status_message_.c_str());

        ImGui::SameLine(io.DisplaySize.x - 150);
        if (renderer_) {
            const char* mode = renderer_->GetCameraController().IsTransitioning() ? "Transitioning" : "Free-fly";
            ImGui::Text("Mode: %s", mode);
        }
    }
    ImGui::End();
}

}  // namespace viewer
}  // namespace psynth
