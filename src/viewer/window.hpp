#pragma once

#include <functional>
#include <string>
#include <vector>

struct GLFWwindow;

namespace psynth {
namespace viewer {

class Window {
public:
    using KeyCallback = std::function<void(int key, int scancode, int action, int mods)>;
    using MouseButtonCallback = std::function<void(int button, int action, int mods)>;
    using CursorPosCallback = std::function<void(double xpos, double ypos)>;
    using ScrollCallback = std::function<void(double xoffset, double yoffset)>;
    using DropCallback = std::function<void(const std::vector<std::string>& paths)>;
    using FramebufferSizeCallback = std::function<void(int width, int height)>;

    Window(int width, int height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool ShouldClose() const;
    void PollEvents();
    void SwapBuffers();

    void BeginFrame();
    void EndFrame();

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    float GetAspectRatio() const { return static_cast<float>(width_) / height_; }

    double GetTime() const;
    float GetDeltaTime() const { return delta_time_; }

    bool IsKeyPressed(int key) const;
    bool IsMouseButtonPressed(int button) const;
    void GetCursorPos(double& x, double& y) const;
    void SetCursorMode(int mode);

    void SetKeyCallback(KeyCallback cb) { key_callback_ = std::move(cb); }
    void SetMouseButtonCallback(MouseButtonCallback cb) { mouse_button_callback_ = std::move(cb); }
    void SetCursorPosCallback(CursorPosCallback cb) { cursor_pos_callback_ = std::move(cb); }
    void SetScrollCallback(ScrollCallback cb) { scroll_callback_ = std::move(cb); }
    void SetDropCallback(DropCallback cb) { drop_callback_ = std::move(cb); }
    void SetFramebufferSizeCallback(FramebufferSizeCallback cb) { framebuffer_size_callback_ = std::move(cb); }

    GLFWwindow* GetHandle() const { return window_; }

private:
    static void KeyCallbackStatic(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void MouseButtonCallbackStatic(GLFWwindow* window, int button, int action, int mods);
    static void CursorPosCallbackStatic(GLFWwindow* window, double xpos, double ypos);
    static void ScrollCallbackStatic(GLFWwindow* window, double xoffset, double yoffset);
    static void DropCallbackStatic(GLFWwindow* window, int count, const char** paths);
    static void FramebufferSizeCallbackStatic(GLFWwindow* window, int width, int height);

    GLFWwindow* window_ = nullptr;
    int width_;
    int height_;
    float delta_time_ = 0.0f;
    double last_frame_time_ = 0.0;

    KeyCallback key_callback_;
    MouseButtonCallback mouse_button_callback_;
    CursorPosCallback cursor_pos_callback_;
    ScrollCallback scroll_callback_;
    DropCallback drop_callback_;
    FramebufferSizeCallback framebuffer_size_callback_;
};

}  // namespace viewer
}  // namespace psynth
