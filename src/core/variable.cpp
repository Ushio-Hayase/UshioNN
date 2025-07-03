#include "core/variable.h"

namespace ushionn
{
Variable::Variable() : data_(), grad_(), type_(DataType::FLOAT32), device_(DataLocation::NONE)
{
}

Variable::Variable(std::vector<size_t> shape) : data_(shape, DataType::FLOAT32), grad_(shape, DataType::FLOAT32)
{
    if (device_ == DataLocation::HOST)
    {
        data_.allocate_cpu_mem(data_.get_total_bytes());
        grad_.allocate_cpu_mem(grad_.get_total_bytes());
    }
    else if (device_ == DataLocation::DEVICE)
    {
        data_.allocate_gpu_mem(data_.get_total_bytes());
        grad_.allocate_gpu_mem(grad_.get_total_bytes());
    }
    else
    {
        USHIONN_LOG_FATAL("Variable의 위치가 NONE입니다.");
    }
}

Variable::Variable(std::vector<size_t> shape, DataType type, DataLocation device)
    : data_(shape, type), grad_(shape, type), type_(type), device_(device)
{
    if (device_ == DataLocation::HOST)
    {
        data_.allocate_cpu_mem(data_.get_total_bytes());
        grad_.allocate_cpu_mem(grad_.get_total_bytes());
    }
    else if (device_ == DataLocation::DEVICE)
    {
        data_.allocate_gpu_mem(data_.get_total_bytes());
        grad_.allocate_gpu_mem(grad_.get_total_bytes());
    }
    else
    {
        USHIONN_LOG_FATAL("Variable의 위치가 NONE입니다.");
    }
}

std::vector<size_t> Variable::get_shape() const
{
    return data_.get_shape();
}

DataType Variable::get_type() const
{
    return type_;
}

DataLocation Variable::get_device() const
{
    return device_;
}

}  // namespace ushionn