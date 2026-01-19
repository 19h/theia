#include <psynth/common.hpp>
#include <psynth/config.hpp>
#include <psynth/features/sift.hpp>
#include <psynth/geometry/fundamental.hpp>
#include <psynth/io/dataset.hpp>
#include <psynth/io/export.hpp>
#include <psynth/io/serialization.hpp>
#include <psynth/matching/matcher.hpp>
#include <psynth/mvs/pmvs_export.hpp>
#include <psynth/sfm/incremental_sfm.hpp>
#include <psynth/sfm/tracks.hpp>
#include <psynth/types.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string images_dir;
  std::string out_dir;
  std::string config_path;
  bool cache = true;
};

void PrintUsage(const char* argv0) {
  std::cerr << "Usage:\n"
            << "  " << argv0
            << " --images_dir <dir> --out_dir <dir> [--config <file.yml>] [--no_cache]\n";
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw psynth::Error("Missing value for " + name);
      return argv[++i];
    };

    if (arg == "--images_dir") {
      a.images_dir = need_value(arg);
    } else if (arg == "--out_dir") {
      a.out_dir = need_value(arg);
    } else if (arg == "--config") {
      a.config_path = need_value(arg);
    } else if (arg == "--no_cache") {
      a.cache = false;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw psynth::Error("Unknown argument: " + arg);
    }
  }

  if (a.images_dir.empty() || a.out_dir.empty()) {
    PrintUsage(argv[0]);
    throw psynth::Error("images_dir and out_dir are required");
  }

  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);

    psynth::PipelineConfig cfg =
        args.config_path.empty() ? psynth::PipelineConfig() : psynth::PipelineConfig::FromYAML(args.config_path);

    psynth::io::ImageDataset dataset = psynth::io::ImageDataset::FromDirectory(args.images_dir, cfg);

    const fs::path out_dir(args.out_dir);
    psynth::io::EnsureDir(out_dir);

    cfg.SaveYAML((out_dir / "effective_config.yml").string());

    // =========================
    // Feature extraction
    // =========================
    std::vector<psynth::FeatureSet> features;
    const fs::path features_dir = out_dir / "features";

    bool have_features = false;
    if (args.cache && fs::exists(features_dir)) {
      have_features = psynth::io::LoadAllFeatures(features_dir, dataset.size(), &features);
    }

    if (!have_features) {
      features.resize(dataset.size());
      psynth::io::EnsureDir(features_dir);

      for (int i = 0; i < dataset.size(); ++i) {
        cv::Mat gray = dataset.ReadGray(i);
        features[i] = psynth::features::ExtractSIFT(gray, cfg.sift);
        std::cerr << "[psynth] features image " << i << " keypoints=" << features[i].keypoints.size() << "\n";
      }

      PSYNTH_REQUIRE(psynth::io::SaveAllFeatures(features_dir, features), "Failed to save features");
    } else {
      std::cerr << "[psynth] loaded cached features from " << features_dir.string() << "\n";
    }

    // =========================
    // Pairwise matching + F RANSAC
    // =========================
    std::vector<psynth::VerifiedPair> verified_pairs;
    const fs::path verified_path = out_dir / "verified_pairs.yml.gz";

    bool have_verified = false;
    if (args.cache && fs::exists(verified_path)) {
      have_verified = psynth::io::LoadVerifiedPairs(verified_path, &verified_pairs);
    }

    if (!have_verified) {
      const int N = dataset.size();

      for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
          const auto matches = psynth::matching::MatchSiftSymmetric(features[i], features[j], cfg.matcher);
          if (static_cast<int>(matches.size()) < cfg.matcher.min_matches) continue;

          std::vector<Eigen::Vector2d> x1;
          std::vector<Eigen::Vector2d> x2;
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
          opt.rng_seed = static_cast<uint32_t>(i * 73856093u) ^ static_cast<uint32_t>(j * 19349663u);

          const auto r = psynth::geometry::EstimateFundamentalRansac(x1, x2, opt);
          if (!r.success) continue;

          psynth::VerifiedPair vp;
          vp.pair.i = i;
          vp.pair.j = j;
          vp.F = r.F;
          vp.num_ransac_iters = r.iterations_run;
          vp.inlier_threshold_px = r.inlier_threshold_px;

          vp.inliers.reserve(r.inlier_indices.size());
          for (const int idx_inlier : r.inlier_indices) vp.inliers.push_back(matches[idx_inlier]);

          verified_pairs.push_back(std::move(vp));

          std::cerr << "[psynth] verified pair (" << i << "," << j << ") inliers=" << r.inlier_indices.size()
                    << " / " << matches.size() << "\n";
        }
      }

      PSYNTH_REQUIRE(psynth::io::SaveVerifiedPairs(verified_path, verified_pairs), "Failed to save verified_pairs");
    } else {
      std::cerr << "[psynth] loaded cached verified pairs from " << verified_path.string() << "\n";
    }

    // =========================
    // Track building
    // =========================
    psynth::sfm::Tracks tracks = psynth::sfm::BuildTracksUnionFind(features, dataset.size(), verified_pairs);
    std::cerr << "[psynth] tracks built: " << tracks.all().size() << "\n";

    // =========================
    // Incremental SfM + BA
    // =========================
    psynth::sfm::IncrementalSfM sfm(dataset, features, verified_pairs, std::move(tracks), cfg);
    psynth::sfm::Reconstruction rec = sfm.Run();

    std::cerr << "[psynth] reconstruction: cameras=" << rec.cameras.size() << " tracks=" << rec.tracks.all().size()
              << "\n";

    // =========================
    // Export
    // =========================
    PSYNTH_REQUIRE(psynth::io::ExportSparsePLY((out_dir / "sparse.ply").string(), rec), "Failed to write sparse.ply");
    PSYNTH_REQUIRE(psynth::io::ExportCamerasPLY((out_dir / "cameras.ply").string(), rec), "Failed to write cameras.ply");
    PSYNTH_REQUIRE(psynth::io::ExportBundleOut((out_dir / "bundle.out").string(), dataset, rec), "Failed to write bundle.out");

    if (cfg.pmvs.enabled) {
      const fs::path pmvs_root = out_dir / cfg.pmvs.pmvs_root;
      PSYNTH_REQUIRE(psynth::mvs::ExportPMVS(pmvs_root.string(), dataset, rec, cfg), "PMVS export failed");
      std::cerr << "[psynth] PMVS input exported to: " << pmvs_root.string() << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[psynth] error: " << e.what() << "\n";
    return 1;
  }
}