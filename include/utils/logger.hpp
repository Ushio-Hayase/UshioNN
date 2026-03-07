//
// Created by UshioHayase on 2025-11-11.
//
#pragma once
#include <cstdint>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <ios>
#include <sstream>
#include <string>

namespace nunet
{
namespace utils
{
enum class LogLevel
{
    Info,
    Warning,
    Error
};

class Logger
{
  private:
#if defined(_WIN32)
    HANDLE file_handle_ = nullptr;
#else
    int file_handle_ = 0;
#endif
    LogLevel min_level_;

  public:
    static Logger& getInstance();

    void setLogLevel(LogLevel lvl);

    // 외부에서 복사/이동을 금지
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    template <typename... Args>
    void write(LogLevel level, const char* file, int line, const char* function,
               const std::string& format, Args... args) const;

  private:
    Logger();
    ~Logger();

    // 재귀 끝
    static std::string formatMessage(const std::string& format);
    // 재귀 호출
    template <typename Value, typename... Rest>
    std::string formatMessage(const std::string& format, Value&& first,
                              Rest&&... rest) const;

    // 최종 로그 문자열을 만드는 함수
    static std::string buildLogEntry(const LogLevel lvl, const char* file,
                                     int line, const char* func,
                                     const std::string& format);

    // 각 채널에 출력하는 함수
    void outputToChannels(const std::string& log) const;
};

void setLogLevel(LogLevel lvl);

template <typename Value, typename... Rest>
std::string Logger::formatMessage(const std::string& format, Value&& first,
                                  Rest&&... rest) const
{
    auto brace_pos = format.find("{}");

    if (brace_pos == std::string::npos)
        return format;

    std::string before = format.substr(0, brace_pos);

    std::string after = format.substr(brace_pos + 2);

    std::stringstream ss;

    if constexpr (std::is_convertible_v<Value, const char*>)
        ss << first;
    else if constexpr (std::is_same_v<std::decay_t<Value>, std::string>)
        ss << first;
    else if constexpr (std::is_pointer_v<std::decay_t<Value>>)
        ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(first)
           << std::dec;
    else
        ss << first;

    return before + ss.str() +
           formatMessage(after, std::forward<Rest>(rest)...);
}

template <typename... Args>
void Logger::write(const LogLevel level, const char* file, int line,
                   const char* function, const std::string& format,
                   Args... args) const
{
    if (level < min_level_)
        return;

    std::string user_message = formatMessage(format, args...);
    std::string final_log =
        buildLogEntry(level, file, line, function, user_message);
    outputToChannels(final_log);
}
} // namespace utils
} // namespace nunet
