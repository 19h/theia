#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace psynth {
namespace viewer {

struct CameraState {
    glm::vec3 position{0.0f, 0.0f, 5.0f};
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    float fov = 60.0f;  // Vertical FOV in degrees

    glm::vec3 GetForward() const;
    glm::vec3 GetRight() const;
    glm::vec3 GetUp() const;
    glm::mat4 GetViewMatrix() const;
    glm::mat4 GetProjectionMatrix(float aspect_ratio) const;
};

class CameraController {
public:
    enum class Mode {
        FreeFly,
        Transitioning
    };

    CameraController();

    void Update(float delta_time, bool capture_input);

    // Free-fly controls
    void ProcessKeyboard(int key, int action);
    void ProcessMouseMovement(float xoffset, float yoffset);
    void ProcessScroll(float yoffset);

    // Teleportation
    void TeleportTo(const CameraState& target, float duration = 0.5f);
    bool IsTransitioning() const { return mode_ == Mode::Transitioning; }

    // State access
    const CameraState& GetState() const { return current_; }
    CameraState& GetState() { return current_; }
    Mode GetMode() const { return mode_; }

    // Settings
    float movement_speed = 5.0f;
    float mouse_sensitivity = 0.1f;
    float scroll_speed = 0.5f;

private:
    void UpdateFreeFly(float delta_time);
    void UpdateTransition(float delta_time);

    // Smooth interpolation helpers
    static float EaseInOutCubic(float t);
    static glm::quat SlerpSafe(const glm::quat& a, const glm::quat& b, float t);

    CameraState current_;
    Mode mode_ = Mode::FreeFly;

    // Free-fly state
    float yaw_ = -90.0f;
    float pitch_ = 0.0f;
    bool move_forward_ = false;
    bool move_backward_ = false;
    bool move_left_ = false;
    bool move_right_ = false;
    bool move_up_ = false;
    bool move_down_ = false;

    // Transition state
    CameraState transition_start_;
    CameraState transition_target_;
    float transition_duration_ = 0.0f;
    float transition_elapsed_ = 0.0f;
};

}  // namespace viewer
}  // namespace psynth
