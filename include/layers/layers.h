#pragma once

#include <memory>  // for std::shared_ptr
#include <random>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "core/variable.h"
#include "layers/activation.h"

namespace ushionn
{
namespace nn
{

class Layer
{
   public:
    Layer(std::string name) : name_(std::move(name)), data_type_(DataType::FLOAT32) {}
    Layer(std::string name, DataType type) : name_(std::move(name)), data_type_(type) {}
    virtual ~Layer();

    /// @brief 레이어의 이름을 반환
    /// @return 레이어의 이름
    const inline std::string& get_name() const { return name_; }

    virtual spVariable forward(spVariable v);
    virtual spVariable forward(spVariable v1, spVariable v2);

    virtual void backward(Tensor& p_grad);

    virtual spVariable forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
    virtual void backward(Tensor& p_grad, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

    // 파라미터에 대한 그래디언트
    virtual void initialize_parameters_norm(unsigned long long seed = 0) = 0;  // 파라미터 초기화

    virtual std::vector<int64_t> get_output_shape(std::vector<int64_t> input_dims) const = 0;

   protected:
    int id;
    std::string name_;
    DataType data_type_ = DataType::FLOAT32;

    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
};

class DenseLayer : public Layer
{
   public:
    DenseLayer();
    DenseLayer(spVariable* w, spVariable* b, bool isTranspose = false);
    DenseLayer(spVariable* w, bool isTranspose = false);
    DenseLayer(int input_size, int output_size);
    DenseLayer(int input_size, int output_size, bool no_bias);

    spVariable forward(std::vector<spVariable>& inputs, std::vector<spVariable>& outputs);
    void backward(Tensor& p_grad, std::vector<spVariable>& inputs, std::vector<spVariable>& outputs);

    void initialize_parameters_norm(unsigned long long seed = 0);

    std::vector<int64_t> get_output_shape(std::vector<int64_t> input_dims) const override;

   private:
    Tensor weights_;
    Tensor bias_;
    Tensor weights_grad_;
    Tensor bias_grad_;
};
class ReLULayer : public Layer
{
   public:
    ReLULayer();
    spVariable forward(std::vector<spVariable>& inputs, std::vector<spVariable>& outputs);
    void backward(Tensor& p_grad, std::vector<spVariable>& inputs, std::vector<spVariable>& outputs);

   private:
    spVariable rr = nullptr;
};
}  // namespace nn
}  // namespace ushionn