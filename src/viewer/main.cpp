#include "window.hpp"
#include "renderer.hpp"
#include "ui/viewer_ui.hpp"

#include <psynth/common.hpp>
#include <psynth/config.hpp>
#include <psynth/features/sift.hpp>
#include <psynth/geometry/fundamental.hpp>
#include <psynth/io/dataset.hpp>
#include <psynth/io/serialization.hpp>
#include <psynth/matching/matcher.hpp>
#include <psynth/sfm/incremental_sfm.hpp>
#include <psynth/sfm/tracks.hpp>

#include <filesystem>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>

namespace {

constexpr int kWindowWidth = 1600;
constexpr int kWindowHeight = 900;
constexpr const char* kWindowTitle = "psynth Viewer";

class ViewerApp {
public:
    ViewerApp() = default;

    bool Initialize() {
        try {
            window_ = std::make_unique<psynth::viewer::Window>(kWindowWidth, kWindowHeight, kWindowTitle);
        } catch (const std::exception& e) {
            std::cerr << "Failed to create window: " << e.what() << std::endl;
            return false;
        }

        if (!renderer_.Initialize(window_->GetWidth(), window_->GetHeight())) {
            std::cerr << "Failed to initialize renderer" << std::endl;
            return false;
        }

        ui_.SetRenderer(&renderer_);

        // Set up callbacks
        window_->SetKeyCallback([this](int key, int scancode, int action, int mods) {
            (void)scancode; (void)mods;
            renderer_.OnKey(key, action);
        });

        window_->SetMouseButtonCallback([this](int button, int action, int mods) {
            (void)mods;
            double x, y;
            window_->GetCursorPos(x, y);
            renderer_.OnMouseButton(button, action, x, y);
        });

        window_->SetCursorPosCallback([this](double x, double y) {
            renderer_.OnMouseMove(x, y);
        });

        window_->SetScrollCallback([this](double xoffset, double yoffset) {
            (void)xoffset;
            renderer_.OnScroll(yoffset);
        });

        window_->SetDropCallback([this](const std::vector<std::string>& paths) {
            ui_.AddDroppedFiles(paths);
        });

        window_->SetFramebufferSizeCallback([this](int width, int height) {
            renderer_.Resize(width, height);
        });

        // Set up UI callbacks
        ui_.SetReconstructCallback([this](const std::vector<std::string>& paths) {
            StartReconstruction(paths);
        });

        ui_.SetLoadCallback([this](const std::string& path) {
            LoadReconstruction(path);
        });

        return true;
    }

    void Run() {
        while (!window_->ShouldClose()) {
            window_->PollEvents();
            window_->BeginFrame();

            // Check for reconstruction completion
            CheckReconstructionProgress();

            bool capture_input = !ui_.WantsCaptureInput();
            renderer_.Update(window_->GetDeltaTime(), capture_input);
            renderer_.Render();

            ui_.Render();

            window_->EndFrame();
            window_->SwapBuffers();
        }
    }

private:
    void StartReconstruction(const std::vector<std::string>& image_paths) {
        if (reconstruction_running_) {
            return;
        }

        if (image_paths.empty()) {
            ui_.SetStatusMessage("No images to reconstruct");
            return;
        }

        reconstruction_running_ = true;
        ui_.SetReconstructionRunning(true);
        ui_.SetStatusMessage("Starting reconstruction...");

        reconstruction_thread_ = std::thread([this, image_paths]() {
            try {
                namespace fs = std::filesystem;
                psynth::PipelineConfig cfg;

                // Determine directory from first image
                fs::path first_image(image_paths[0]);
                fs::path images_dir = first_image.parent_path();

                psynth::io::ImageDataset dataset = psynth::io::ImageDataset::FromDirectory(images_dir.string(), cfg);

                if (dataset.size() == 0) {
                    throw std::runtime_error("No images found in directory");
                }

                // Feature extraction
                std::vector<psynth::FeatureSet> features(dataset.size());
                for (int i = 0; i < dataset.size(); ++i) {
                    cv::Mat gray = dataset.ReadGray(i);
                    features[i] = psynth::features::ExtractSIFT(gray, cfg.sift);
                    std::cerr << "[viewer] features image " << i << " keypoints=" << features[i].keypoints.size() << "\n";
                }

                // Pairwise matching
                std::vector<psynth::VerifiedPair> verified_pairs;
                const int N = dataset.size();

                for (int i = 0; i < N; ++i) {
                    for (int j = i + 1; j < N; ++j) {
                        const auto matches = psynth::matching::MatchSiftSymmetric(features[i], features[j], cfg.matcher);
                        if (static_cast<int>(matches.size()) < cfg.matcher.min_matches) continue;

                        std::vector<Eigen::Vector2d> x1, x2;
                        x1.reserve(matches.size());
                        x2.reserve(matches.size());

                        for (const auto& m : matches) {
                            const auto& kp1 = features[i].keypoints[m.kp1];
                            const auto& kp2 = features[j].keypoints[m.kp2];
                            x1.emplace_back(kp1.pt.x, kp1.pt.y);
                            x2.emplace_back(kp2.pt.x, kp2.pt.y);
                        }

                        psynth::geometry::FundamentalRansacOptions opt;
                        opt.max_iterations = cfg.fund_ransac.max_iterations;
                        opt.confidence = cfg.fund_ransac.confidence;
                        opt.inlier_threshold_px = cfg.fund_ransac.inlier_threshold_px;
                        opt.min_inliers = cfg.fund_ransac.min_inliers;

                        const auto r = psynth::geometry::EstimateFundamentalRansac(x1, x2, opt);
                        if (!r.success) continue;

                        psynth::VerifiedPair vp;
                        vp.pair.i = i;
                        vp.pair.j = j;
                        vp.F = r.F;
                        vp.num_ransac_iters = r.iterations_run;
                        vp.inlier_threshold_px = r.inlier_threshold_px;

                        vp.inliers.reserve(r.inlier_indices.size());
                        for (const int idx : r.inlier_indices) vp.inliers.push_back(matches[idx]);

                        verified_pairs.push_back(std::move(vp));
                        std::cerr << "[viewer] pair (" << i << "," << j << ") inliers=" << r.inlier_indices.size() << "\n";
                    }
                }

                // Track building
                psynth::sfm::Tracks tracks = psynth::sfm::BuildTracksUnionFind(features, dataset.size(), verified_pairs);
                std::cerr << "[viewer] tracks built: " << tracks.all().size() << "\n";

                // Incremental SfM
                psynth::sfm::IncrementalSfM sfm(dataset, features, verified_pairs, std::move(tracks), cfg);
                psynth::sfm::Reconstruction rec = sfm.Run();

                {
                    std::lock_guard<std::mutex> lock(reconstruction_mutex_);
                    pending_reconstruction_ = std::make_unique<psynth::sfm::Reconstruction>(std::move(rec));
                    pending_images_ = dataset.images();
                    reconstruction_success_ = true;
                }
            } catch (const std::exception& e) {
                std::cerr << "Reconstruction failed: " << e.what() << std::endl;
                reconstruction_success_ = false;
            }

            reconstruction_running_ = false;
        });
    }

    void CheckReconstructionProgress() {
        if (!reconstruction_running_ && reconstruction_thread_.joinable()) {
            reconstruction_thread_.join();
            ui_.SetReconstructionRunning(false);

            std::lock_guard<std::mutex> lock(reconstruction_mutex_);
            if (reconstruction_success_ && pending_reconstruction_) {
                // Debug: count triangulated tracks
                int triangulated_count = 0;
                for (const auto& track : pending_reconstruction_->tracks.all()) {
                    if (track.triangulated) triangulated_count++;
                }
                std::cerr << "[viewer] DEBUG: total tracks=" << pending_reconstruction_->tracks.all().size()
                          << " triangulated=" << triangulated_count << std::endl;

                renderer_.LoadReconstruction(*pending_reconstruction_, pending_images_);
                ui_.SetStatusMessage("Reconstruction complete: " +
                                     std::to_string(triangulated_count) + " points, " +
                                     std::to_string(pending_reconstruction_->cameras.size()) + " cameras");
                pending_reconstruction_.reset();
                pending_images_.clear();
            } else {
                ui_.SetStatusMessage("Reconstruction failed");
            }
        }
    }

    void LoadReconstruction(const std::string& path) {
        // TODO: Implement reconstruction loading from file
        // The serialization module doesn't currently support full reconstruction loading
        (void)path;
        ui_.SetStatusMessage("Loading from file not yet supported");
    }

    std::unique_ptr<psynth::viewer::Window> window_;
    psynth::viewer::Renderer renderer_;
    psynth::viewer::ViewerUI ui_;

    // Reconstruction threading
    std::thread reconstruction_thread_;
    std::atomic<bool> reconstruction_running_{false};
    std::atomic<bool> reconstruction_success_{false};
    std::mutex reconstruction_mutex_;
    std::unique_ptr<psynth::sfm::Reconstruction> pending_reconstruction_;
    std::vector<psynth::ImageInfo> pending_images_;
};

}  // namespace

int main(int argc, char** argv) {
    ViewerApp app;

    if (!app.Initialize()) {
        return 1;
    }

    // Optionally load reconstruction from command line
    if (argc > 1) {
        // Could be an image directory or a reconstruction file
        std::string arg = argv[1];
        // For now, just print usage
        std::cout << "Usage: psynth_viewer [reconstruction.json]" << std::endl;
        std::cout << "Or drag and drop images into the window to start reconstruction." << std::endl;
    }

    app.Run();

    return 0;
}
