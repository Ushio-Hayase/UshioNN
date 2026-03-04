//
// Created by UshioHayase on 2025-11-11.
//

#include "utils/logger.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <format>

namespace nunet
{
namespace utils
{
Logger::Logger() : min_level_(LogLevel::Info)
{
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::floor<std::chrono::seconds>(now);
    const auto local_time =
        std::chrono::zoned_time{std::chrono::current_zone(), t};

    if (!std::filesystem::exists("./logs"))
        CreateDirectoryW(L"./logs", nullptr);

    const std::wstring file_name =
        std::format(L"./logs/Log_{:%Y-%m-%d_%H-%M-%S}.log", local_time);
    file_handle_ = CreateFileW(file_name.c_str(), GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                               CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);

#if defined(DEBUG) || defined(_DEBUG)
    console_out_handle_ = GetStdHandle(STD_OUTPUT_HANDLE);
    console_err_handle_ = GetStdHandle(STD_ERROR_HANDLE);
#endif
}

Logger::~Logger() { CloseHandle(file_handle_); }

Logger& Logger::getInstance()
{
    static Logger instance;
    return instance;
}

void Logger::setLogLevel(LogLevel lvl) { min_level_ = lvl; }

void setLogLevel(LogLevel lvl) { Logger::getInstance().setLogLevel(lvl); }

std::string Logger::formatMessage(const std::string& format) { return format; }

std::string Logger::buildLogEntry(LogLevel lvl, const char* file, int line,
                                  const char* func, const std::string& format)
{
    std::stringstream ss;

    // 1. 현재 시간 구하기 (std::chrono)
    const auto now = std::chrono::system_clock::now();
    const auto time_t = std::chrono::system_clock::to_time_t(now);

    tm local_time;
    localtime_s(&local_time, &time_t);

    // 시간 포맷팅: [YYYY-MM-DD HH:MM:SS]
    ss << "[" << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << "]";

    // 로그 레벨 표기
    switch (lvl)
    {
    case LogLevel::Info:
        ss << " [INFO] ";
        break;
    case LogLevel::Warning:
        ss << " [WARN] ";
        break;
    case LogLevel::Error:
        ss << " [ERROR]  ";
        break;
    }

    // 파일, 줄, 함수 정보 표기
    ss << "[" << file << ":" << line << ", " << func << "] ";

    ss << format;

    return ss.str();
}

void Logger::outputToChannels(const std::string& log) const
{
    constexpr char NEXT_LINE = '\n';
    const std::string msg = log + NEXT_LINE;

    if (file_handle_ != nullptr)
        WriteFile(file_handle_, msg.c_str(), msg.size(), nullptr, nullptr);

    if (console_out_handle_ == nullptr)
        return;
#if defined(DEBUG) || defined(_DEBUG)
    WriteConsole(console_out_handle_, msg.c_str(), msg.length(), nullptr,
                 nullptr);
    WriteConsole(console_out_handle_, &NEXT_LINE, 1, nullptr, nullptr);
#endif
}
} // namespace utils
} // namespace nunet
