#include "camera_controller.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace psynth {
namespace viewer {

glm::vec3 CameraState::GetForward() const {
    return orientation * glm::vec3(0.0f, 0.0f, -1.0f);
}

glm::vec3 CameraState::GetRight() const {
    return orientation * glm::vec3(1.0f, 0.0f, 0.0f);
}

glm::vec3 CameraState::GetUp() const {
    return orientation * glm::vec3(0.0f, 1.0f, 0.0f);
}

glm::mat4 CameraState::GetViewMatrix() const {
    glm::vec3 forward = GetForward();
    glm::vec3 up = GetUp();
    return glm::lookAt(position, position + forward, up);
}

glm::mat4 CameraState::GetProjectionMatrix(float aspect_ratio) const {
    return glm::perspective(glm::radians(fov), aspect_ratio, 0.1f, 1000.0f);
}

CameraController::CameraController() {
    // Initialize orientation from yaw/pitch
    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front = glm::normalize(front);

    glm::vec3 world_up(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(front, world_up));
    glm::vec3 up = glm::normalize(glm::cross(right, front));

    glm::mat3 rotation_matrix(right, up, -front);
    current_.orientation = glm::quat_cast(rotation_matrix);
}

void CameraController::Update(float delta_time, bool capture_input) {
    switch (mode_) {
        case Mode::FreeFly:
            if (capture_input) {
                UpdateFreeFly(delta_time);
            }
            break;
        case Mode::Transitioning:
            UpdateTransition(delta_time);
            break;
    }
}

void CameraController::ProcessKeyboard(int key, int action) {
    bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    bool released = (action == GLFW_RELEASE);

    if (key == GLFW_KEY_W) {
        if (pressed) move_forward_ = true;
        if (released) move_forward_ = false;
    }
    if (key == GLFW_KEY_S) {
        if (pressed) move_backward_ = true;
        if (released) move_backward_ = false;
    }
    if (key == GLFW_KEY_A) {
        if (pressed) move_left_ = true;
        if (released) move_left_ = false;
    }
    if (key == GLFW_KEY_D) {
        if (pressed) move_right_ = true;
        if (released) move_right_ = false;
    }
    if (key == GLFW_KEY_SPACE) {
        if (pressed) move_up_ = true;
        if (released) move_up_ = false;
    }
    if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
        if (pressed) move_down_ = true;
        if (released) move_down_ = false;
    }
}

void CameraController::ProcessMouseMovement(float xoffset, float yoffset) {
    if (mode_ != Mode::FreeFly) return;

    xoffset *= mouse_sensitivity;
    yoffset *= mouse_sensitivity;

    yaw_ += xoffset;
    pitch_ += yoffset;

    pitch_ = std::clamp(pitch_, -89.0f, 89.0f);

    // Update orientation from yaw/pitch
    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front = glm::normalize(front);

    glm::vec3 world_up(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(front, world_up));
    glm::vec3 up = glm::normalize(glm::cross(right, front));

    glm::mat3 rotation_matrix(right, up, -front);
    current_.orientation = glm::quat_cast(rotation_matrix);
}

void CameraController::ProcessScroll(float yoffset) {
    movement_speed += yoffset * scroll_speed;
    movement_speed = std::clamp(movement_speed, 0.5f, 50.0f);
}

void CameraController::TeleportTo(const CameraState& target, float duration) {
    if (mode_ == Mode::Transitioning) return;

    transition_start_ = current_;
    transition_target_ = target;
    transition_duration_ = duration;
    transition_elapsed_ = 0.0f;
    mode_ = Mode::Transitioning;
}

void CameraController::UpdateFreeFly(float delta_time) {
    float velocity = movement_speed * delta_time;
    glm::vec3 forward = current_.GetForward();
    glm::vec3 right = current_.GetRight();
    glm::vec3 up(0.0f, 1.0f, 0.0f);  // World up for vertical movement

    if (move_forward_) current_.position += forward * velocity;
    if (move_backward_) current_.position -= forward * velocity;
    if (move_left_) current_.position -= right * velocity;
    if (move_right_) current_.position += right * velocity;
    if (move_up_) current_.position += up * velocity;
    if (move_down_) current_.position -= up * velocity;
}

void CameraController::UpdateTransition(float delta_time) {
    transition_elapsed_ += delta_time;
    float t = transition_elapsed_ / transition_duration_;

    if (t >= 1.0f) {
        current_ = transition_target_;
        mode_ = Mode::FreeFly;

        // Update yaw/pitch from the target orientation
        glm::vec3 forward = current_.GetForward();
        pitch_ = glm::degrees(asin(forward.y));
        yaw_ = glm::degrees(atan2(forward.z, forward.x));
        return;
    }

    float eased_t = EaseInOutCubic(t);

    current_.position = glm::mix(transition_start_.position, transition_target_.position, eased_t);
    current_.orientation = SlerpSafe(transition_start_.orientation, transition_target_.orientation, eased_t);
    current_.fov = glm::mix(transition_start_.fov, transition_target_.fov, eased_t);
}

float CameraController::EaseInOutCubic(float t) {
    return t < 0.5f
        ? 4.0f * t * t * t
        : 1.0f - pow(-2.0f * t + 2.0f, 3.0f) / 2.0f;
}

glm::quat CameraController::SlerpSafe(const glm::quat& a, const glm::quat& b, float t) {
    // Handle the case where quaternions are nearly opposite
    glm::quat b_adjusted = b;
    if (glm::dot(a, b) < 0.0f) {
        b_adjusted = -b;
    }
    return glm::slerp(a, b_adjusted, t);
}

}  // namespace viewer
}  // namespace psynth
