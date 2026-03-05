//
// Created by UshioHayase on 2025-11-11.
//
#pragma once

#include "utils/logger.hpp"

namespace nunet
{
namespace utils
{
#if defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)
#define LOG_INFO(format, ...)                                                  \
    ::nunet::utils::Logger::getInstance().write(                               \
        ::nunet::utils::LogLevel::Info, __FILE__, __LINE__, __FUNCTION__,      \
        format, ##__VA_ARGS__)

#define LOG_WARN(format, ...)                                                  \
    ::nunet::utils::Logger::getInstance().write(                               \
        ::nunet::utils::LogLevel::Warning, __FILE__, __LINE__, __FUNCTION__,   \
        format, ##__VA_ARGS__)

#define LOG_ERROR(format, ...)                                                 \
    ::nunet::utils::Logger::getInstance().write(                               \
        ::nunet::utils::LogLevel::Error, __FILE__, __LINE__, __FUNCTION__,     \
        format, ##__VA_ARGS__)

#define ASSERT(condition)                                                      \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            LOG_ERROR("Assertion Failed: {}", #condition);                     \
            __debugbreak();                                                    \
        }                                                                      \
    } while (false)

#define ASSERT_EQ(val1, val2)                                                  \
    do                                                                         \
    {                                                                          \
        if (!((val1) == (val2)))                                               \
        {                                                                      \
            LOG_ERROR("Assertion Failed: {} == {}. val1: {}, val2: {}", #val1, \
                      #val2, (val1), (val2));                                  \
            __debugbreak();                                                    \
        }                                                                      \
    } while (false)

#define ASSERT_NE(val1, val2)                                                  \
    do                                                                         \
    {                                                                          \
        if ((val1) == (val2))                                                  \
        {                                                                      \
            LOG_ERROR("Assertion Failed : {} != {}. val1: {}, val2: {}",       \
                      #val1, #val2, (val1), (val2));                           \
            __debugbreak();                                                    \
        }                                                                      \
    } while (false)

#define ASSERT_MESSAGE(condition, message)                                     \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            LOG_ERROR(#message);                                               \
            __debugbreak();                                                    \
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
#endif
} // namespace utils
} // namespace nunet