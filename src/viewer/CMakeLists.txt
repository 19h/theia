# Viewer library and executable

include(FetchContent)

# Fetch Dear ImGui from GitHub
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.6
)
FetchContent_MakeAvailable(imgui)

set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(GLAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glad)
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/stb)

# ImGui sources
set(IMGUI_SOURCES
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Glad sources
set(GLAD_SOURCES
  ${GLAD_DIR}/glad.c
)

# Viewer library
add_library(psynth_viewer_lib
  window.cpp
  renderer.cpp
  camera_controller.cpp
  point_cloud_renderer.cpp
  frustum_renderer.cpp
  photo_compositor.cpp
  shader.cpp
  ui/viewer_ui.cpp
  ${IMGUI_SOURCES}
  ${GLAD_SOURCES}
)

target_include_directories(psynth_viewer_lib
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GLAD_DIR}
    ${STB_DIR}
)

target_link_libraries(psynth_viewer_lib
  PUBLIC
    psynth
    glfw
    OpenGL::GL
    glm::glm
)

# Link macOS frameworks for stb_image
if(APPLE)
  target_link_libraries(psynth_viewer_lib
    PUBLIC
      "-framework CoreFoundation"
      "-framework CoreGraphics"
      "-framework ImageIO"
  )
endif()

target_compile_definitions(psynth_viewer_lib
  PRIVATE
    IMGUI_IMPL_OPENGL_LOADER_GLAD
)

# Copy shaders to build directory
file(GLOB SHADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag)
foreach(SHADER ${SHADER_FILES})
  get_filename_component(SHADER_NAME ${SHADER} NAME)
  configure_file(${SHADER} ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME} COPYONLY)
endforeach()

# Viewer executable
add_executable(psynth_viewer main.cpp)
target_link_libraries(psynth_viewer PRIVATE psynth_viewer_lib)
