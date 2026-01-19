#pragma once

#include <glm/glm.hpp>
#include <string>

namespace psynth {
namespace viewer {

class Shader {
public:
    Shader() = default;
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;
    Shader(Shader&& other) noexcept;
    Shader& operator=(Shader&& other) noexcept;

    bool LoadFromFile(const std::string& vertex_path, const std::string& fragment_path);
    bool LoadFromSource(const std::string& vertex_source, const std::string& fragment_source);

    void Use() const;

    void SetBool(const std::string& name, bool value) const;
    void SetInt(const std::string& name, int value) const;
    void SetFloat(const std::string& name, float value) const;
    void SetVec2(const std::string& name, const glm::vec2& value) const;
    void SetVec3(const std::string& name, const glm::vec3& value) const;
    void SetVec4(const std::string& name, const glm::vec4& value) const;
    void SetMat3(const std::string& name, const glm::mat3& value) const;
    void SetMat4(const std::string& name, const glm::mat4& value) const;

    unsigned int GetID() const { return id_; }
    bool IsValid() const { return id_ != 0; }

private:
    unsigned int id_ = 0;

    static bool CompileShader(unsigned int shader, const std::string& source);
    static bool LinkProgram(unsigned int program);
};

}  // namespace viewer
}  // namespace psynth
