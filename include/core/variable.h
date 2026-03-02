#pragma once

#include "core/tensor.h"

namespace ushionn
{

static int gspVariable_id = 500;

class Variable
{
   public:
    Variable();

    Variable(std::vector<size_t> shape);
    Variable(std::vector<size_t> shape, DataType type = DataType::FLOAT32,
             DataLocation device = DataLocation::HOST);

    const Tensor& get_data_ref() const;
    Tensor& get_data_ref_mutable();

    const Tensor& get_grad_ref() const;
    Tensor& get_grad_ref_mutable();

    std::vector<size_t> get_shape() const;
    DataType get_type() const;
    DataLocation get_device() const;

   private:
    Tensor data_;
    Tensor grad_;

    int id_ = gspVariable_id++;
    bool is_last_backward_ = false;
    DataType type_;
    DataLocation device_;
};

using spVariable = std::shared_ptr<Variable>;
}  // namespace ushionn
