//
// Created by UshioHayase on 2026-03-14.
//

#pragma once
#include <driver_types.h>

namespace ushionn
{
namespace Accelerator
{
class CudaStream
{
  public:
    void synchronize();

  private:
    cudaStream_t stream_;
};
} // namespace Accelerator
} // namespace ushionn
