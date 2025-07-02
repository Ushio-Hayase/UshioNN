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

spVariable ReLULayer::forward(std::vector<spVariable>& inputs, std::vector<spVariable>& outputs)
{
    spVariable x = inputs.at(0);
    spVariable result = spVariable(new Variable(x->get_shape()));

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