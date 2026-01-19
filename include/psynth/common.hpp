#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace psynth {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

[[noreturn]] inline void Throw(const char* file, int line, const std::string& msg) {
  std::ostringstream oss;
  oss << file << ":" << line << ": " << msg;
  throw Error(oss.str());
}

}  // namespace psynth

#define PSYNTH_REQUIRE(cond, msg)      \
  do {                                \
    if (!(cond)) {                    \
      ::psynth::Throw(__FILE__, __LINE__, (msg)); \
    }                                 \
  } while (0)