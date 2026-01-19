#pragma once

#include <functional>
#include <string>
#include <vector>

namespace psynth {
namespace viewer {

class Renderer;

class ViewerUI {
public:
    using ReconstructCallback = std::function<void(const std::vector<std::string>& paths)>;
    using LoadCallback = std::function<void(const std::string& path)>;
    using ExportCallback = std::function<void(const std::string& path)>;

    ViewerUI();
    ~ViewerUI();

    void SetRenderer(Renderer* renderer) { renderer_ = renderer; }

    void SetReconstructCallback(ReconstructCallback cb) { reconstruct_callback_ = std::move(cb); }
    void SetLoadCallback(LoadCallback cb) { load_callback_ = std::move(cb); }
    void SetExportCallback(ExportCallback cb) { export_callback_ = std::move(cb); }

    void Render();

    // For drag-and-drop
    void AddDroppedFiles(const std::vector<std::string>& paths);

    // Status
    void SetStatusMessage(const std::string& msg) { status_message_ = msg; }
    void SetReconstructionRunning(bool running) { reconstruction_running_ = running; }

    bool WantsCaptureInput() const { return wants_capture_input_; }

private:
    void RenderMenuBar();
    void RenderSidePanel();
    void RenderStatusBar();
    void RenderDropZone();

    Renderer* renderer_ = nullptr;

    ReconstructCallback reconstruct_callback_;
    LoadCallback load_callback_;
    ExportCallback export_callback_;

    std::vector<std::string> dropped_files_;
    std::string status_message_ = "Ready";
    bool reconstruction_running_ = false;
    bool wants_capture_input_ = false;

    // UI state
    bool show_demo_window_ = false;
};

}  // namespace viewer
}  // namespace psynth
