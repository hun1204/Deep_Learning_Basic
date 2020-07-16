# 21강 시작!
# NAND OR AND 조합 없이 딥러닝을 이용한 XOR 문제 해결!!
# XOR의 입력 데이터 [[0,0],[0,1],[1,0],[1,1]]일때 t는 [0,1,1,0]

# XOR 문제 - 딥러닝 아키텍처
# 입력층과 은닉층 사이의 가중치와 은닉층과 출력층 사이의 가중치를 각각
# W(2),W(3)으로 가정하고 입력되는 값은 x1 x2 입력층은 입력데이터 개수에 맞게 생성
# 출력층은 1개의 노드이며 은닉층에 대한 노드개수에 제한은 없으면 임의로 만들 수 있다.

# 딥러닝에서는, 1개 이상의 은닉 층(hidden layer)을 만들 수 있고, 각 은닉 층(hidden layer)에
# 존재하는 노드(node) 개수 또한 임의의 개수를 만들 수 있음, 그러나 은닉 층과 노드 수가 많아지면
# 학습 속도가 느려지므로 적절한 개수의 은닉 층과 노드 수를 고려하여 구현 하는 것이 필요함

# 코드예제!
# LogicGate class - 딥러닝 버전

# external function
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(x))
def numerical_derivative(f,x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x,flags=["multi_index"],op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx]= float(tmp_val) + delta_x
        fx1 = f(x)
        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x)
        
        grad[idx]= (fx1-fx2)/(2*delta_x)
        
        x[idx] = tmp_val
        it.iternext()

    return grad
    
# LogicGate class
class LogicGate:

    def __init__(self, gate_name,xdata,tdata):
        self.name = gate_name
        
        #입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4,2) #4개의 입력데이터 x1, x2에 대하여 batch 처리행렬(여기서 batch란? 데이터를 일괄적으로 모아서 처리하는 작업)
        self.__tdata = tdata.reshape(4,1) #4개의 입력데이터 x1, x2 에 대한 각각의 계산 값 행렬
        
        #2층 hidden layer unit : 6개 가정, 가중치 W2, 바이어스 b2 초기화
        self.__W2 = np.random.rand(2,6)
        self.__b2 = np.random.rand(6)
        
        #3층 output layer unit : 1개, 가중치W2, 바이어스 b3 초기화
        self.__W3 = np.random.rand(6,1)
        self.__b3 = np.random.rand(1)
        
        #학습률 learning rate 초기화
        self.__learning_rate = 1e-2

        print(self.name + " objcet is created")

    def feed_forward(self): # feed forward를 통하여 손실함수(cross-entropy) 값 계산
        delta = 1e-7 # log 무한대 발산 방지
        
        z2 = np.dot(self.__xdata, self.__W2) + self.__b2 # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)    #은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3  # 출력층의 선형회귀 값
        a3 = sigmoid(z3)    #출력층의 출력
        y = a3

        #cross-entropy
        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))
    
    def loss_val(self): 
        delta = 1e-7 # log 무한대 발산 방지
        
        z2 = np.dot(self.__xdata, self.__W2) + self.__b2 # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)    #은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3  # 출력층의 선형회귀 값
        a3 = sigmoid(z3)    #출력층의 출력
        y= a3
        
        #cross-entropy
        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))
    
    #수치미분을 이용하여 손실함수가 최소가 될때 까지 학습하는 함수
    def train(self):
        f = lambda x : self.feed_forward()
        print("initial loss value = ", self.loss_val())
        for step in range(10001):
            self.__W2 -= self.__learning_rate * numerical_derivative(f, self.__W2)
            self.__b2 -= self.__learning_rate * numerical_derivative(f, self.__b2)
            self.__W3 -= self.__learning_rate * numerical_derivative(f, self.__W3)
            self.__b3 -= self.__learning_rate * numerical_derivative(f, self.__b3)
            if(step%400 == 0):
                print("step = ", step," , loss value = ", self.loss_val())

    def predict(self, xdata):
        z2 = np.dot(xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)
        
        z3 = np.dot(a2, self.__W3) + self.__b3
        a3 = sigmoid(z3)
        y = a3

        if y > 0.5:
            result = 1 #True
        else:
            result = 0 #False

        return y, result

# AND GATE 객체 생성 및 training
xdata = np.array([[0,0],[0,1],[1,0],[1,1]])
tdata = np.array([0,0,0,1])

and_obj = LogicGate("AND_GATE",xdata,tdata)
and_obj.train()

test_data = xdata
for data in test_data:
    print(and_obj.predict(data))

# OR GATE 객체 생성 및 TRAINING
tdata = np.array([0,1,1,1])
OR_obj = LogicGate("OR_GATE",xdata,tdata)
OR_obj.train()

test_data = xdata
for data in test_data:
    print(OR_obj.predict(data))

# XOR GATE 객체 생성 및 TRAINING
tdata = np.array([0,1,1,0])
XOR_obj = LogicGate("XOR_GATE",xdata,tdata)
XOR_obj.train()

test_data = xdata
for data in test_data:
    print(XOR_obj.predict(data))

# 신경망 기반의 딥러닝읠 구현하여 XOR문제 해결!!
# 1. NAND / OR / AND 조합을 이용하지 않고,
# 2. 입력층 / 은닉층 / 출력층으로 구성된 딥러닝 아키텍처(Neural Network) 이용하여

# 입력층, 1개 이상의 은닉층, 출력층을 가지는 딥러닝을 설계한다면, 이런 딥러닝을
# 이용해서 XOR 보다는 더 복잡한 필기체 손 글씨 인식, 이미지 인식 등의 문제도 해결할 수
# 있다는 insight를 얻을 수 있다.
