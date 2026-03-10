#pragma once
// 필요하다면 여기에 라이브러리 자체 에러 코드 열거형 등을 추가할 수 있습니다.
namespace ushionn
{
enum class NUnetError
{
    SUCCESS = 0,
    GENERAL_ERROR,
    CUDA_ERROR,
};
} // namespace nunet