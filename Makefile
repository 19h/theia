# psynth Makefile - wraps CMake for convenience

BUILD_DIR := build
BUILD_TYPE ?= Release
CMAKE_FLAGS ?=
JOBS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Executables
VIEWER := $(BUILD_DIR)/src/viewer/psynth_viewer
PIPELINE := $(BUILD_DIR)/psynth_pipeline

.PHONY: all build configure clean rebuild viewer pipeline run help install

# Default target
all: build

# Configure CMake
configure:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS)

# Build everything
build: configure
	@cmake --build $(BUILD_DIR) -j$(JOBS)

# Build only the viewer
viewer: configure
	@cmake --build $(BUILD_DIR) --target psynth_viewer -j$(JOBS)

# Build only the pipeline CLI
pipeline: configure
	@cmake --build $(BUILD_DIR) --target psynth_pipeline -j$(JOBS)

# Build only the library
lib: configure
	@cmake --build $(BUILD_DIR) --target psynth -j$(JOBS)

# Run the viewer
run: viewer
	@$(VIEWER)

# Run the pipeline (pass ARGS for arguments)
run-pipeline: pipeline
	@$(PIPELINE) $(ARGS)

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)

# Full rebuild
rebuild: clean build

# Install (requires sudo for system-wide)
install: build
	@cmake --install $(BUILD_DIR)

# Debug build
debug:
	@$(MAKE) BUILD_TYPE=Debug build

# Release build with optimizations
release:
	@$(MAKE) BUILD_TYPE=Release build

# Build without viewer
no-viewer:
	@$(MAKE) CMAKE_FLAGS="-DPSYNTH_BUILD_VIEWER=OFF" build

# Format code (if clang-format is available)
format:
	@find src include apps -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format -i 2>/dev/null || echo "clang-format not found"

# Show help
help:
	@echo "psynth build system"
	@echo ""
	@echo "Usage: make [target] [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  all, build    Build everything (default)"
	@echo "  viewer        Build only the viewer"
	@echo "  pipeline      Build only the CLI pipeline"
	@echo "  lib           Build only the library"
	@echo "  run           Build and run the viewer"
	@echo "  run-pipeline  Build and run pipeline (use ARGS=\"...\")"
	@echo "  clean         Remove build directory"
	@echo "  rebuild       Clean and build"
	@echo "  configure     Run CMake configuration only"
	@echo "  install       Install to system"
	@echo "  debug         Build with debug symbols"
	@echo "  release       Build with optimizations"
	@echo "  no-viewer     Build without the viewer"
	@echo "  format        Format source code"
	@echo "  help          Show this help"
	@echo ""
	@echo "Options:"
	@echo "  BUILD_TYPE=Debug|Release  Set build type (default: Release)"
	@echo "  JOBS=N                    Parallel jobs (default: auto)"
	@echo "  CMAKE_FLAGS=\"...\"         Additional CMake flags"
	@echo "  ARGS=\"...\"                Arguments for run-pipeline"
	@echo ""
	@echo "Examples:"
	@echo "  make                      # Build everything"
	@echo "  make run                  # Build and launch viewer"
	@echo "  make debug                # Debug build"
	@echo "  make rebuild JOBS=4       # Clean rebuild with 4 jobs"
	@echo "  make run-pipeline ARGS=\"/path/to/images\""
