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
