//
// Created by UshioHayase on 2025-11-11.
//
#pragma once

#include <cassert>

#include "utils/logger.hpp"

namespace ushionn::utils
{

#if defined(_MSC_VER)
// Windows MSVC
#define DEBUG_BREAK() __debugbreak()
#elif defined(__clang__)
// Linux/macOS Clang
#define DEBUG_BREAK() __builtin_debugtrap()
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
// Linux GCC (x86/x64)
#define DEBUG_BREAK() __asm__ volatile("int $0x03")
#else
// Fallback (Other Linux/Unix architectures)
#include <signal.h>
#define DEBUG_BREAK() raise(SIGTRAP)
#endif

#if defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)
#define LOG_INFO(format, ...)                                                  \
    ::ushionn::utils::Logger::getInstance().write(                             \
        ::ushionn::utils::LogLevel::Info, __FILE__, __LINE__, __FUNCTION__,    \
        format, ##__VA_ARGS__)

#define LOG_WARN(format, ...)                                                  \
    ::ushionn::utils::Logger::getInstance().write(                             \
        ::ushionn::utils::LogLevel::Warning, __FILE__, __LINE__, __FUNCTION__, \
        format, ##__VA_ARGS__)

#define LOG_ERROR(format, ...)                                                 \
    ::ushionn::utils::Logger::getInstance().write(                             \
        ::ushionn::utils::LogLevel::Error, __FILE__, __LINE__, __FUNCTION__,   \
        format, ##__VA_ARGS__)

#define ASSERT(condition)                                                      \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            LOG_ERROR("Assertion Failed: {}", #condition);                     \
            DEBUG_BREAK();                                                     \
            assert(!(condition));                                              \
        }                                                                      \
    } while (false)

#define ASSERT_EQ(val1, val2)                                                  \
    do                                                                         \
    {                                                                          \
        if (((val1) != (val2)))                                                \
        {                                                                      \
            LOG_ERROR("Assertion Failed: {} == {}. val1: {}, val2: {}", #val1, \
                      #val2, (val1), (val2));                                  \
            DEBUG_BREAK();                                                     \
            assert((val1) != (val2));                                          \
        }                                                                      \
    } while (false)

#define ASSERT_NE(val1, val2)                                                  \
    do                                                                         \
    {                                                                          \
        if ((val1) == (val2))                                                  \
        {                                                                      \
            LOG_ERROR("Assertion Failed : {} != {}. val1: {}, val2: {}",       \
                      #val1, #val2, (val1), (val2));                           \
            DEBUG_BREAK();                                                     \
            assert(!((val1) == (val2)));                                       \
        }                                                                      \
    } while (false)

#define ASSERT_MESSAGE(condition, message, ...)                                \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            LOG_ERROR(message, ##__VA_ARGS__);                                 \
            DEBUG_BREAK();                                                     \
            assert(!(condition));                                              \
        }                                                                      \
    } while (false);

#else
#define LOG_INFO(format, ...) (void(0))
#define LOG_WARN(format, ...) (void(0))
#define LOG_ERROR(format, ...)                                                 \
    ::ushionn::utils::Logger::getInstance().write(                             \
        ::ushionn::utils::LogLevel::Error, __FILE__, __LINE__, __FUNCTION__,   \
        format, ##__VA_ARGS__)
#define ASSERT(condition) (void(0))
#define ASSERT_EQ(val1, val2) (void(0))
#define ASSERT_NE(val1, val2) (void(0))
#define ASSERT_MESSAGE(condition, message) (void(0))
#endif
} // namespace ushionn::utils
