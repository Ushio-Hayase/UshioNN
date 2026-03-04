# Function 클래스와 Context 클래스 구조

## Context 클래스 역할

순전파 시점의 중간 데이터를 역전파 시점까지 유지하는 데이터 저장소.

## Function 클래스 메서드

- forward(): 순전파 수학 연산을 수행하고, 역전파에 필요한 데이터를 Context의 save_for_backward()를 통해 저장.

- backward(): Context의 get_saved_tensors()를 통해 데이터를 불러와 편미분을 계산.
