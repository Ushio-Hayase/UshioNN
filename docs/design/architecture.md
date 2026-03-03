# UshioNN 아키텍처

## 디렉토리 구조

- core
- cuda
- layers
- model

## 모듈 역할

- core : 핵심 코어로 다른 모듈들에서 공통적으로 사용하는 코어 클래스 파일
- cuda : cuda를 이용하는 함수가 써진 cu파일들과 헤더들
- layers : 신경망의 레이어 클래스 파일들
- model : 신경망의 모델을 빌드할 때 쓰는 모델 클래스 파일

## 클래스 구조

```mermaid
classDiagram
    class Tensor {
        -data: float*
        -grad: float*
        -shape: vector~int~
        -device: Device
        -grad_fn: shared_ptr~Function~
        +Tensor(shape, device)
        +backward() void
        +zero_grad() void
        +matmul(Tensor other) Tensor
    }

    class Function {
        <<interface>>
        +forward(Context* ctx, vector~Tensor~ inputs) Tensor
        +backward(Context* ctx, Tensor grad_output) vector~Tensor~
    }
    
    class MatmulFunction {
        +forward(Context* ctx, vector~Tensor~ inputs) Tensor
        +backward(Context* ctx, Tensor grad_output) vector~Tensor~
    }
    
    class AddFunction {
        +forward(Context* ctx, vector~Tensor~ inputs) Tensor
        +backward(Context* ctx, Tensor grad_output) vector~Tensor~
    }

    class Module {
        <<abstract>>
        #_parameters: map~string, Tensor~
        #_modules: map~string, shared_ptr~Module~~
        +forward(Tensor input)* Tensor
        +parameters() vector~Tensor~
        +to(Device target_device) void
        +add_module(string name, shared_ptr~Module~ module) void
    }

    class Linear {
        -weight: Tensor
        -bias: Tensor
        +Linear(int in_features, int out_features)
        +forward(Tensor input) Tensor
    }

    class ReLU {
        +forward(Tensor input) Tensor
    }

    class Model {
        -layers: vector~shared_ptr~Module~~
        +Model()
        +forward(Tensor input) Tensor
    }

    %% Relationships
    Tensor "1" *-- "0..1" Function : created_by (grad_fn)
    Function <|-- MatmulFunction : implements
    Function <|-- AddFunction : implements
    Function ..> Tensor : depends on
    
    Module <|-- Linear : inherits
    Module <|-- ReLU : inherits
    Module <|-- Model : inherits
    
    Module "1" *-- "many" Tensor : manages parameters
    Model "1" *-- "many" Module : composite layers
```

## 텐서플로 or 파이토치 작동원리

텐서플로나 파이토치는 자동미분 엔진을 가지고 있음.
이 라이브러리들의 자동미분 엔진은 각 연산들을 추적하고 기록하여 역전파를 쉽게 만들어줌.
자동 미분을 사용하면 신경망의 순전파 단계에서 연산 그래프를 생성함.
이 그래프의 노드는 텐서이고 엣지는 입력 텐서로부터 새로운 텐서를 만드는 함수임.

```python
import torch
import math

# 입력값과 출력값을 갖는 텐서들을 생성합니다.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 이 예제에서, 출력 y는 (x, x^2, x^3)의 선형 함수이므로, 선형 계층 신경망으로 간주할 수 있습니다.
# (x, x^2, x^3)를 위한 텐서를 준비합니다.
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# 위 코드에서, x.unsqueeze(-1)은 (2000, 1)의 shape을, p는 (3,)의 shape을 가지므로,
# 이 경우 브로드캐스트(broadcast)가 적용되어 (2000, 3)의 shape을 갖는 텐서를 얻습니다.

# nn 패키지를 사용하여 모델을 순차적 계층(sequence of layers)으로 정의합니다.
# nn.Sequential은 다른 Module을 포함하는 Module로, 포함되는 Module들을 순차적으로 적용하여 
# 출력을 생성합니다. 각각의 Linear Module은 선형 함수(linear function)를 사용하여 입력으로부터
# 출력을 계산하고, 내부 Tensor에 가중치와 편향을 저장합니다.
# Flatten 계층은 선형 계층의 출력을 `y` 의 shape과 맞도록(match) 1D 텐서로 폅니다(flatten).
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# 또한 nn 패키지에는 주로 사용되는 손실 함수(loss function)들에 대한 정의도 포함되어 있습니다;
# 여기에서는 평균 제곱 오차(MSE; Mean Squared Error)를 손실 함수로 사용하겠습니다.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):

    # 순전파 단계: x를 모델에 전달하여 예측값 y를 계산합니다. Module 객체는 __call__ 연산자를 
    # 덮어써서(override) 함수처럼 호출할 수 있도록 합니다. 이렇게 함으로써 입력 데이터의 텐서를 Module에 전달하여
    # 출력 데이터의 텐서를 생성합니다.
    y_pred = model(xx)

    # 손실을 계산하고 출력합니다. 예측한 y와 정답인 y를 갖는 텐서들을 전달하고,
    # 손실 함수는 손실(loss)을 갖는 텐서를 반환합니다.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계를 실행하기 전에 변화도(gradient)를 0으로 만듭니다.
    model.zero_grad()

    # 역전파 단계: 모델의 학습 가능한 모든 매개변수에 대해 손실의 변화도를 계산합니다.
    # 내부적으로 각 Module의 매개변수는 requires_grad=True일 때 텐서에 저장되므로,
    # 아래 호출은 모델의 모든 학습 가능한 매개변수의 변화도를 계산하게 됩니다.
    loss.backward()

    # 경사하강법을 사용하여 가중치를 갱신합니다.
    # 각 매개변수는 텐서이므로, 이전에 했던 것처럼 변화도에 접근할 수 있습니다.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# list의 첫번째 항목에 접근하는 것처럼 `model` 의 첫번째 계층(layer)에 접근할 수 있습니다.
linear_layer = model[0]

# 선형 계층에서, 매개변수는 `weights` 와 `bias` 로 저장됩니다.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

```
- [pytorch 공식 예제](https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html#pytorch-nn)

또는 객체지향적으로 각 라이브러리의 기초 모델 클래스를 상속하는 형태로도 만들 수 있음. 

```python
import random
import torch
import math


class DynamicNet(torch.nn.Module):
    def __init__(self):
        """
        생성자에서 5개의 매개변수를 생성(instantiate)하고 멤버 변수로 지정합니다.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        모델의 순전파 단계에서는 무작위로 4, 5 중 하나를 선택한 뒤 매개변수 e를 재사용하여
        이 차수들의의 기여도(contribution)를 계산합니다.

        각 순전파 단계는 동적 연산 그래프를 구성하기 때문에, 모델의 순전파 단계를 정의할 때
        반복문이나 조건문과 같은 일반적인 Python 제어-흐름 연산자를 사용할 수 있습니다.

        여기에서 연산 그래프를 정의할 때 동일한 매개변수를 여러번 사용하는 것이 완벽히 안전하다는
        것을 알 수 있습니다.
        """
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        """
        Python의 다른 클래스(class)처럼, PyTorch 모듈을 사용해서 사용자 정의 메소드를 정의할 수 있습니다.
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'


# 입력값과 출력값을 갖는 텐서들을 생성합니다.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 위에서 정의한 클래스로 모델을 생성합니다.
model = DynamicNet()

# 손실 함수와 optimizer를 생성합니다. 이 이상한 모델을 순수한 확률적 경사하강법(SGD; Stochastic Gradient Descent)으로
# 학습하는 것은 어려우므로, 모멘텀(momentum)을 사용합니다.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(30000):
    # 순전파 단계: 모델에 x를 전달하여 예측값 y를 계산합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력합니다.
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())

    # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
```

