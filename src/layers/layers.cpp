#include "layers/layers.h"

namespace ushionn
{
namespace nn
{

spVariable Layer::forward(spVariable v)
{
    inputs.emplace_back(std::move(v));

    spVariable result = forward(inputs, outputs);

    return result;
}

spVariable Layer::forward(spVariable v1, spVariable v2)
{
    inputs.emplace_back(std::move(v1));
    inputs.emplace_back(std::move(v2));

    spVariable result = forward(inputs, outputs);

    return result;
}

void Layer::backward(Tensor& p_grad)
{
    backward(p_grad, inputs, outputs);
}

DenseLayer::DenseLayer()
{
    weights_ = new Variable();  // 스마트 포인터 사용
    bias_ = new Variable();
}

DenseLayer::DenseLayer(Variable* w, Variable* b, bool isTranspose) : isTranspose_(isTranspose)
{
    this->weights_ = w;
    this->bias_ = b;
}

DenseLayer::DenseLayer(Variable* w, bool isTranspose) : isTranspose_(isTranspose)
{
    this->weights_ = w;
    this->bias_ = nullptr;
}

DenseLayer::~DenseLayer()
{
    delete weights_;
    delete bias_;
}

spVariable DenseLayer::forward(std::vector<spVariable>& inputs, std::vector<spVariable>& outputs)
{
    spVariable x = inputs.at(0);

    // 올바른 출력 차원 계산
    std::vector<size_t> output_shape = {x->get_shape()[x->get_data_ref().get_shape_size() - 2],
                                        weights_->get_shape().at(weights_->get_data_ref().get_shape_size() - 1)};
    spVariable result = spVariable(new Variable(output_shape, weights_->get_type(), weights_->get_device()));

    Tensor tmp(output_shape, x->get_type());
    tmp.to(x->get_device());

    x->get_data_ref().dot(weights_->get_data_ref(), tmp);  // x * W

    if (bias_ != nullptr)
    {
        result->get_data_ref_mutable() = tmp + bias_->get_data_ref();
    }
    else
    {
        result->get_data_ref_mutable() = std::move(tmp);
    }

    return result;
}

void DenseLayer::backward(Tensor& p_grad, std::vector<spVariable>& inputs, std::vector<spVariable>& outputs)
{
    spVariable x = inputs.at(0);
    x->get_grad_ref_mutable() += weights_->get_data_ref().transpose().dot(p_grad);
    weights_->get_grad_ref_mutable() += p_grad.dot(x->get_data_ref().transpose());

    // 편향 그래디언트 계산
    if (bias_ != nullptr)
    {
        // 배치 크기만큼 1로 채워진 벡터 생성
        std::vector<size_t> ones_shape = {p_grad.get_shape()[0]};  // 배치 크기
        Tensor ones(ones_shape, p_grad.get_type());

        if (p_grad.get_device() == DataLocation::DEVICE)
        {
            ones.allocate_gpu_mem(ones.get_total_bytes());
            ones.to(DataLocation::DEVICE);
            // GPU에서 1로 초기화하는 커널 호출 필요
        }
        else
        {
            ones.allocate_cpu_mem(ones.get_total_bytes());
            ones.to(DataLocation::HOST);

            if (p_grad.get_type() == DataType::FLOAT32)
            {
                float* ones_data = static_cast<float*>(ones.get_cpu_ptr_mutable());
                std::fill(ones_data, ones_data + ones_shape[0], 1.0f);
            }
            else if (p_grad.get_type() == DataType::FLOAT64)
            {
                double* ones_data = static_cast<double*>(ones.get_cpu_ptr_mutable());
                std::fill(ones_data, ones_data + ones_shape[0], 1.0);
            }
        }

        // ones^T * p_grad = 배치 차원에 대한 합
        Tensor bias_grad_sum = ones.transpose().dot(p_grad);
        bias_->get_grad_ref_mutable() += bias_grad_sum;
    }
}

spVariable ReLULayer::forward(std::vector<spVariable>& inputs, std::vector<spVariable>& outputs)
{
    spVariable x = inputs.at(0);
    spVariable result = spVariable(new Variable(x->get_shape(), x->get_type(), x->get_device()));

    relu(x->get_data_ref(), result->get_data_ref_mutable());

    return result;
}

void ReLULayer::backward(Tensor& p_grad, std::vector<spVariable>& inputs, std::vector<spVariable>& outputs)
{
    spVariable x = inputs.at(0);
    Tensor tmp(x->get_shape(), x->get_type());

    if (x->get_device() == DataLocation::HOST)
    {
        tmp.to(DataLocation::HOST);
        relu_d(x->get_grad_ref(), tmp);
        x->get_grad_ref_mutable() += tmp * p_grad;
    }
    else if (x->get_device() == DataLocation::DEVICE)
    {
        tmp.to(DataLocation::DEVICE);
        relu_d(x->get_grad_ref(), tmp);
        x->get_grad_ref_mutable() += tmp * p_grad;
    }
}

}  // namespace nn
}  // namespace ushionn