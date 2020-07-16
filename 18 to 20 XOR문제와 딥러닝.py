# 18강 시작!
# AND, OR, NAND, XOR 논리테이블(Logic Table)은 입력데이터 (x1,x2), 정답데이터 t (0 또는 1)
# 인 머신러닝 Training Data와 개념적으로 동일함
# -> 즉, 논리게이트는 손실함수로 cross-entropy를 이용해서 Logistic Regression (Classification)
# 알고리즘으로 데이터를 분류하고 결과를 예측할 수 있음

# AND연산을 살펴보면
# X1, X2 값이 REGRESSION X*W+b를 만들고 sigmoid 함수를 이용해 y = sigmoid(z) 식이 만들어진다.

# LogicGate Class - AND, OR, NAND, XOR 검증방법
import numpy as np

# 1. external function
def sigmoid(x): # 0또는 1을 출력하기 위한 sigmoid 함수
    return 1/(1+np.exp(-x))
def numerical_derivative(f,x): # 수치미분함수
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val)+delta_x
        fx1 = f(x)
        x[idx] = float(tmp_val)-delta_x
        fx2 = f(x)

        grad[idx] = (fx1-fx2) / (2*delta_x)
        x[idx] = tmp_val
        it.iternext()
    
    return grad

# 2.LogicGate Class
class LogicGate:
    def __init__(self, gate_name, xdata, tdata): #___xdata, __tdata, __W, __b 초기화
        self.name = gate_name

        #이제부터 모두 __으로 private 선언
        # 입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4,2)
        self.__tdata = tdata.reshape(4,1)

        # 가중치 W, 바이어스 b 초기화
        self.__W = np.random.rand(2,1)
        self.__b = np.random.rand(1)

        # 학습율 learning_rate 초기화 
        # 만약 손실함수계산중에 최소값이 안나오고 발산하면 더 작은값으로 세팅해줘야 함
        self.__learning_rate = 1e-2

    # 손실함수
    def __loss_func(self):  # 손실함수 cross-entropy

        delta = 1e-7 # log 값 무한대 발산 방지

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        #cross-entropy
        return -np.sum(self.__tdata*np.log(y+delta) + (1- self.__tdata)*np.log((1-y)+delta))

    def error_val(self): # 손실함수 값 계산
        delta = 1e-7 # log 값 무한대 발산 방지

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        #cross-entropy
        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))

    def train(self): # 수치미분을 이용하여 손실함수 최소값 찾는 method
        f = lambda x : self.__loss_func()
        print("Initial error value = ", self.error_val())
        for step in range(8001):
            self.__W -= self.__learning_rate * numerical_derivative(f, self.__W)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)
            if (step % 100 == 0):
                print("step = ", step, "error value = ", self.error_val())
    def predict(self, input_data): # 미래값 예측 method
        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)

        if y > 0.5:
            result = 1 #True
        else:
            result = 0 #False

        return y, result
# 3.usage
# AND Gate 검증
xdata = np.array([[0,0],[0,1],[1,0],[1,1]]) #입력데이터 생성
tdata = np.array([0,0,0,1]) #정답 데이터 생성 (AND 예시)

AND_obj = LogicGate("AND_GATE",xdata,tdata) #LogicGate 객체생성
AND_obj.train() #손실함수 최소값 갖도록 학습

#임의 데이터에 대해 결과 예측
print(AND_obj.name, "\n")
test_data = np.array([[0,0],[0,1],[1,0],[1,1]])
for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print(input_data, "=",logical_val,"\n")

#OR Gate 검증
xdata = np.array([[0,0],[0,1],[1,0],[1,1]]) #입력데이터 생성
tdata = np.array([0,1,1,1]) #정답 데이터 생성 (or 예시)
OR_obj = LogicGate("OR_GATE",xdata,tdata)
OR_obj.train()

print(OR_obj.name,"\n")
test_data = np.array([[0,0],[0,1],[1,0],[1,1]])
for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data)
    print(sigmoid_val,"    ",input_data, "=",logical_val,"\n")

# NAND Gate 검증
xdata = np.array([[0,0],[0,1],[1,0],[1,1]]) #입력데이터 생성
tdata = np.array([1,1,1,0]) #정답 데이터 생성 (AND 예시)

NAND_obj = LogicGate("NAND_GATE",xdata,tdata) #LogicGate 객체생성
NAND_obj.train() #손실함수 최소값 갖도록 학습

#임의 데이터에 대해 결과 예측
print(NAND_obj.name, "\n")
test_data = np.array([[0,0],[0,1],[1,0],[1,1]])
for input_data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(input_data)
    print(input_data, "=",logical_val,"\n")

#XOR Gate 검증
xdata = np.array([[0,0],[0,1],[1,0],[1,1]]) #입력데이터 생성
tdata = np.array([0,1,1,0]) #정답 데이터 생성 (or 예시)
XOR_obj = LogicGate("OR_GATE",xdata,tdata)
XOR_obj.train()

print(XOR_obj.name,"\n")
test_data = np.array([[0,0],[0,1],[1,0],[1,1]])
for input_data in test_data:
    (sigmoid_val, logical_val) = XOR_obj.predict(input_data)
    print(sigmoid_val,"    ",input_data, "=",logical_val,"\n")

# XOR GATE는 LOGISTIC REGRESSION 알고리즘으로는 데이터를 분류할 수 없어보인다..!!
# 0,1 이 들어간 각각의 입력값에 NAND와 OR을 한 데이터 생성후 두 부류의 데이터에
# AND 연산을 가하면 XOR 출력이 가능하다!!

input_data = np.array([[0,0],[0,1],[1,0],[1,1]])

s1 = []  # NAND 출력
s2 = []  # OR 출력

new_input_data = [] # AND 입력
fianl_output = [] # AND 출력

for index in range(len(input_data)):

    s1 = NAND_obj.predict(input_data[index]) # NAND 출력
    s2 = OR_obj.predict(input_data[index])

    new_input_data.append(s1[-1])
    new_input_data.append(s2[-1])
    
    (sigmoid_val, logical_val) = AND_obj.predict(np.array(new_input_data))

    fianl_output.append(logical_val)
    new_input_data = []

for index in range(len(input_data)):
    print(input_data[index], "=", fianl_output[index], end='')
    print('\n')

# 머신러닝 XOR 문제는 다양한 Gate 조합인 Multi-Layer로 해결할 수 있음
# Layer : 데이터를 처리하거나 계산이 이루어 지는 단위
# 각각의 Gate(NAND, OR, AND)는 Logistic Regression(Classification) 시스템으로 구성됨
# 이전 Gate 모든 출력은 (previous output) 다음 Gate 입력 (next input)으로 들어감

# => insight => 신경망(Neural Network)기반의 딥러닝(Deep Learning)의 핵심 아이디어!!!


# 19강 시작!!
# Review - XOR문제
# 1개의 LOGOISTIC REGRESSION 으로 구현된 XOR은 잘 작동하지않음!
# 따라서 여러개의 Logistic Regression으로 구현된 XOR은 잘 작동한다.
# 이러한 과정은 인간의 신경세포 뉴런(neuron)으로 연결된 신경망 동작원리와 유사함!

# 신경망(Neural Network) - concept(1)
# >신경 세포 뉴런(neuron)은 이전 뉴런으로부터 입력신호를 받아 또 다른 신호를 발생시킨다.
# >그러나 입력에 비례해서 출력을 나타내는 형태 (y=Wx)가 아니라, 입력 값들의 모든 합이
# 어느 임계점(threshold)에 도달해야만 출력 신호를 발생시킨다.
# >이처럼 입력신호를 받아 특정 값의 임계점을 넘어서는 경우에, 출력을 생성해주는 함수를
# 활성화 함수(activation function)라고 하는데, 지금까지 사용해왔던 Logistic Regression 시스템의
# sigmoid 함수가 대표적인 활성화함수이다. 즉, sigmoid에서의 임계점은 0.5로서, 입력값 합이
# 0.5 보다 크다면 1을 출력으로 내보내고, 0.5 보다 값이 작으면 출력을 내보내지 않는다고 볼 수 있다.
# (0은 출력이 없는 상태)
# 활성화함수의 종류 - Sigmoid, ReLU, Leaky ReLU, tangent(tanh)...

# 신경망(Neural Network) - concept(2)
# 인간의 신경 세포인 뉴런 동작원리를 머신러닝에 적용하기 위해서는,
# 1. 입력 신호와 가중치를 곱하고 적당한 바이어스를 더한 후(Linear Regression)
# 2. 이 값을 활성화 함수(sigmoid) 입력으로 전달(Classification) 해서 sigmoid함수
# 임계점 0.5를 넘으면 1을, 아니면 0을 다음 뉴런으로 전달해주어
# multi-variable Logistic Regression 시스템을 구축한다.
# ->입력값이 여러개의 logistic regression을 거쳐서 출력값을 생성함

# 딥러닝(Deep Learning) - concept
# 노드(node) : 1개의 logistic regression을 나타냄
# 노드가 연결되어 있는 신경망 구조를 바탕으로 입력층(Input Layer), 
# 1개 이상의 은닉층(Hidden Layer), 출력층(Output Layer)을 구축하고, 출력층(Output Layer)에서의
# 오차를 기반으로 각 노드(뉴런)의 가중치(Weight)를 학습하는 머신러닝 한 분야

# 참고) 딥러닝 구조에서 1개 이상의 은닉층(hidden layer)을 이용하여 학습시키면 정확도가
# 높은 결과를 얻을 수 있다고 알려져 있음, 즉 은닉층을 깊게(Deep) 할수록 정확도가 높아진다고
# 해서 딥(Deep)러닝이라는 용어가 사용되고 있다.

# 가중치 W21 => 특정 계층의 노드 1에서 다음계층의 노드 2로 전달 되는 신호를 강화 또는
# 약화 시키는 가중치(즉, 다음계층의 노드 번호가 먼저 나옴)
# 만약 Wmn이면 n에서 m으로 전달되는 신호를 강화 약화시키는 가중치!

# 이러한 가중치 값들은, 층과 층사이의 모든 노드에 초기화 되어 있으며, 데이터가 입력층에서
# 출력층으로 전파(propagation) 될 때, 각 층에 있는 노드의 모든 가중치(W11,W12,...Wlk등)는
# 신호를 약화시키거나(낮은 가중치) 또는 신호를 강화(높은 가중치)시키며, 최종적으로는
# 오차가 최소 값이 될 때 최적의 값을 가지게 됨.

# 20강 시작!
# 신경망에서 데이터를 입력층부터 최종 출력으로 전달하는 방식인 feed forward와
# 출력층에서 손실함수 값을 바탕으로 신경망에 가중치 w와 바이어스 b를 업데이트 하는방법에 대해 알아볼예정

# 아키텍처 비교 - logistic regression vs. deep learning
# 1. logistic regression은 input 데이터를 regression으로 바꿔주고
# classification sigmoid(z)를 거쳐서 결과값을 출력함 output
# 2. 신경망을 기반으로 한 Deep learning은 입력층 한개, 여러개의 은닉층, 출력층 한개로
# 각각의 층에는 내부적으로 logistic regression이 구현된 여러개의 노드들로 구현되어있다.
# 은닉층에서의 이전층의 출력이 다음층의 입력으로 들어가는 구조로 되어 있으며
# 이처럼 데이터가 지속적으로 전달되는 흐름을 feed foward 방식이라고 한다!

# 피드포워드 (feed forward) - 표기법(notation)
# 가장 기본적으로 3개의 층과 각 층에 두개의 노드로 구성된 신경망을 생각해보자.
# 1. 계층간 가중치 표기법 (weight notation)
# 가중치 w(2)21 => 계층 2의 노드에 적용되는 가중치로서, 1계층의 노드 1에서 2계층의 노드 2로
# 전달되는 신호를 강화 또는 약화시키는 가중치(즉, 가중치에서의 아래숫자는 다음계층의 노드 번호가 먼저 나옴)
# 2. 노드의 바이어스 표기법 (bias notation)
# 바이어스 b(2)1 => 계층 2에 있는 첫번째 노드(node 1)에 적용되는 바이어스
# 3. 노드의 선형회귀 계산 값 표기법 (linear regression notation)
# 선형회귀 계산값 z(2)2 => 계층 2의 두번째 노드(node 2) 선형회귀 계산 값
# 풀어쓰면 (z(2)2 = x1w(2)21 + x2w(2)22 + b(2)2) 가 된다.
# 4. 노드의 출력 표기법 (node output notation)
# 노드의 출력 값 a(2)2 => 계층 2의 두번째 노드 (node 2) 출력값으로서, logistic regression 계산 값.
# 활성화함수(activation function)로서 sigmoid를 사용한다면 a(2)2 = sigmoid(z(2)2)가 된다.

# 피드포워드 (feed forward) - 동작방식(1)
# 피드포워드란? 입력층(input layer)으로 데이터가 입력되고, 1개 이상으로 구성되는
# 은닉층(hidden layer)을 거쳐서 마지막에 있는 출력 층(output layer)으로 출력 값을 내보내는 과정
# 딥러닝에서는 이전 층(pervious layer)에서 나온 출력 값 => 층과 층 사이에 적용되는
# 가중치(weight) 영향을 받은 다음 => 다음 층(next layer)의 입력 값으로 들어가는 것을 의미함

# 입력 층(input layer) 출력이란? 딥러닝 입력 층에서는 활성화 함수인 sigmoid를
# 적용하지 않고, 입력 값 그대로 출력으로 내보내는 것이 관례화 되어 있음.
# 예시) a(1)1 = x1, a(1)2 = x2 행렬식으로 표현하면 (a(1)1 a(1)2) = (x1 x2)

# 피드포워드 (feed forward) - 동작방식(2)
# 은닉 층(hidden layer)에서의 선형회귀값 z1 z2는 각각
# z(2)1 = a(1)1*W(2)11 + a(1)2*W(2)12 + b(2)1
# z(2)2 = a(1)1*W(2)21 + a(1)2*W(2)22 + b(2)2
# 이러한 값을 행렬식으로 바꿔준다! (1*2 a입력층값)dot(2*2 W가중치값)+(1*2 b바이어스)
# 은닉 층(hidden layer) 출력
# a(2)1 = sigmoid(z(2)1)

# 피드포워드 (feed forward) - 동작방식(3)
# 츨력층(output layer) 선형회귀 값
# 일반식 : z(3)1 = a(2)1*w(3)11 + a(2)2*w(3)12 + b(3)1
# 행렬식 : (z(3)1) = (1*2의 a은닉층값)dot(2*1 W가중치값)+(1*1 b바이어스)

# 출력 층 (output layer) 출력
# y = a(3)1 = sigmoid(z(3)1)
# 출력값 a1은, 입력 데이터에 대해 최종적으로 계산해야 하는 y값 이며, 이러한 y값과
# 정답 t 와의 차이인 오차(loss)를 통해서 가중치와 바이어스를 학습해야 하는것을 의미함

# ***즉 딥러닝 에서는 출력 층(output layer)에서의 출력값(y)과 정답(t)과의 차이를 이용하여,
# 오차가 최소가 되도록 각 층에 있는 가중치와 바이어스를 최적화 해야 함

# 딥러닝 에서의 [W,b] 계산 프로세스
# 1.input 2.feed forward 3.y(E(W,b)가 최소값이면 학습종료 아니면 4.수행) 
# 4.update W(2),b(2),W(3),b(3) => repeat(1.)
