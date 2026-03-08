//
// Created by UshioHayase on 3/8/2026.
//

namespace nunet
{
void cudaDeleter(void* ptr)
{
    if (ptr)
        cudaFree(ptr);
}

} // namespace nunet
