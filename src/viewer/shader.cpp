#include "shader.hpp"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

namespace psynth {
namespace viewer {

Shader::~Shader() {
    if (id_ != 0) {
        glDeleteProgram(id_);
    }
}

Shader::Shader(Shader&& other) noexcept : id_(other.id_) {
    other.id_ = 0;
}

Shader& Shader::operator=(Shader&& other) noexcept {
    if (this != &other) {
        if (id_ != 0) {
            glDeleteProgram(id_);
        }
        id_ = other.id_;
        other.id_ = 0;
    }
    return *this;
}

bool Shader::LoadFromFile(const std::string& vertex_path, const std::string& fragment_path) {
    std::ifstream vert_file(vertex_path);
    std::ifstream frag_file(fragment_path);

    if (!vert_file.is_open()) {
        std::cerr << "Failed to open vertex shader: " << vertex_path << std::endl;
        return false;
    }
    if (!frag_file.is_open()) {
        std::cerr << "Failed to open fragment shader: " << fragment_path << std::endl;
        return false;
    }

    std::stringstream vert_stream, frag_stream;
    vert_stream << vert_file.rdbuf();
    frag_stream << frag_file.rdbuf();

    return LoadFromSource(vert_stream.str(), frag_stream.str());
}

bool Shader::LoadFromSource(const std::string& vertex_source, const std::string& fragment_source) {
    unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);

    if (!CompileShader(vertex_shader, vertex_source)) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }

    if (!CompileShader(fragment_shader, fragment_source)) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }

    unsigned int program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);

    if (!LinkProgram(program)) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        glDeleteProgram(program);
        return false;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    if (id_ != 0) {
        glDeleteProgram(id_);
    }
    id_ = program;

    return true;
}

void Shader::Use() const {
    glUseProgram(id_);
}

void Shader::SetBool(const std::string& name, bool value) const {
    glUniform1i(glGetUniformLocation(id_, name.c_str()), static_cast<int>(value));
}

void Shader::SetInt(const std::string& name, int value) const {
    glUniform1i(glGetUniformLocation(id_, name.c_str()), value);
}

void Shader::SetFloat(const std::string& name, float value) const {
    glUniform1f(glGetUniformLocation(id_, name.c_str()), value);
}

void Shader::SetVec2(const std::string& name, const glm::vec2& value) const {
    glUniform2fv(glGetUniformLocation(id_, name.c_str()), 1, glm::value_ptr(value));
}

void Shader::SetVec3(const std::string& name, const glm::vec3& value) const {
    glUniform3fv(glGetUniformLocation(id_, name.c_str()), 1, glm::value_ptr(value));
}

void Shader::SetVec4(const std::string& name, const glm::vec4& value) const {
    glUniform4fv(glGetUniformLocation(id_, name.c_str()), 1, glm::value_ptr(value));
}

void Shader::SetMat3(const std::string& name, const glm::mat3& value) const {
    glUniformMatrix3fv(glGetUniformLocation(id_, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::SetMat4(const std::string& name, const glm::mat4& value) const {
    glUniformMatrix4fv(glGetUniformLocation(id_, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
}

bool Shader::CompileShader(unsigned int shader, const std::string& source) {
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "Shader compilation failed:\n" << info_log << std::endl;
        return false;
    }
    return true;
}

bool Shader::LinkProgram(unsigned int program) {
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        std::cerr << "Program linking failed:\n" << info_log << std::endl;
        return false;
    }
    return true;
}

}  // namespace viewer
}  // namespace psynth
