#include "window.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <stdexcept>

namespace psynth {
namespace viewer {

Window::Window(int width, int height, const std::string& title)
    : width_(width), height_(height) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);  // Enable vsync

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        glfwDestroyWindow(window_);
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, KeyCallbackStatic);
    glfwSetMouseButtonCallback(window_, MouseButtonCallbackStatic);
    glfwSetCursorPosCallback(window_, CursorPosCallbackStatic);
    glfwSetScrollCallback(window_, ScrollCallbackStatic);
    glfwSetDropCallback(window_, DropCallbackStatic);
    glfwSetFramebufferSizeCallback(window_, FramebufferSizeCallbackStatic);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    last_frame_time_ = glfwGetTime();
}

Window::~Window() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (window_) {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

bool Window::ShouldClose() const {
    return glfwWindowShouldClose(window_);
}

void Window::PollEvents() {
    glfwPollEvents();
}

void Window::SwapBuffers() {
    glfwSwapBuffers(window_);
}

void Window::BeginFrame() {
    double current_time = glfwGetTime();
    delta_time_ = static_cast<float>(current_time - last_frame_time_);
    last_frame_time_ = current_time;

    glfwGetFramebufferSize(window_, &width_, &height_);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Window::EndFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

double Window::GetTime() const {
    return glfwGetTime();
}

bool Window::IsKeyPressed(int key) const {
    return glfwGetKey(window_, key) == GLFW_PRESS;
}

bool Window::IsMouseButtonPressed(int button) const {
    return glfwGetMouseButton(window_, button) == GLFW_PRESS;
}

void Window::GetCursorPos(double& x, double& y) const {
    glfwGetCursorPos(window_, &x, &y);
}

void Window::SetCursorMode(int mode) {
    glfwSetInputMode(window_, GLFW_CURSOR, mode);
}

void Window::KeyCallbackStatic(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self && self->key_callback_) {
        self->key_callback_(key, scancode, action, mods);
    }
}

void Window::MouseButtonCallbackStatic(GLFWwindow* window, int button, int action, int mods) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self && self->mouse_button_callback_) {
        self->mouse_button_callback_(button, action, mods);
    }
}

void Window::CursorPosCallbackStatic(GLFWwindow* window, double xpos, double ypos) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self && self->cursor_pos_callback_) {
        self->cursor_pos_callback_(xpos, ypos);
    }
}

void Window::ScrollCallbackStatic(GLFWwindow* window, double xoffset, double yoffset) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self && self->scroll_callback_) {
        self->scroll_callback_(xoffset, yoffset);
    }
}

void Window::DropCallbackStatic(GLFWwindow* window, int count, const char** paths) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self && self->drop_callback_) {
        std::vector<std::string> path_list;
        path_list.reserve(count);
        for (int i = 0; i < count; ++i) {
            path_list.emplace_back(paths[i]);
        }
        self->drop_callback_(path_list);
    }
}

void Window::FramebufferSizeCallbackStatic(GLFWwindow* window, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self) {
        self->width_ = width;
        self->height_ = height;
        if (self->framebuffer_size_callback_) {
            self->framebuffer_size_callback_(width, height);
        }
    }
}

}  // namespace viewer
}  // namespace psynth
