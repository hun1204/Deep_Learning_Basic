# 23강 시작
# 수치미분으로 가중치/바이어스 업데이트 시 많은 시간이 소요 됨
# 예) 784X100X10 딥러닝 아키텍처에서 60,000 만개의 MNIST를 학습할 경우,
# 컴퓨터 환경에 따라서 10시간 이상 소요됨!!

# 오차역전파 (Back Propagation) - 개념 및 원리
# W(2) = W(2) - 알파*E를 W(2)로 미분한 값 *체인룰에 의해  1ㅡ2ㅡ3으로 분리 (체인 룰은 머신러닝 9강 참고)
# 가중치나 바이어스가 변할 때 최종오차가 얼마나 변하는지를 나타내는 E를 W(2)로
# 미분한 값 같은 편미분을 체인룰을 이용하여 위 식의 1,2,3처럼 국소(LOCAL) 미분으로
# 분리한 후에, 이러한 국소 미분을 수학 공식으로 나타내서 계산하는 방법을 오차역전파
# 라고 한다. (수치미분을 사용않고 행렬로 표현되는 수학공식을 계산하기 때문에 빠른계산 가능)
# 각층의 선형회귀값 Z와 각층의 출력값 A값 계산 아래 링크 참고
# https://www.youtube.com/watch?v=FmVh2qrevOQ&list=PLS8gIc2q83OjStGjdTF2LZtc0vefCAbnX&index=23 참고
# sigmoid(z) = 1 / 1 + e(-z)승을 z값으로 편미분하면 아래와 같은 식이 도출됨
# sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
# 위의 식을 통해 E1을 W(3)11로 편미분한 값을 변형할 수 있음 영상참고
# W(2) 가중치 4번의 편미분 W(3) 가중치 4번의 편미분 b(2)와 b(3)각각의 편미분
# 총 12번의 미분을 업데이트를 해주는 작업이 오래걸리므로 이거를 오차역전파로 구현해야함

# 24강 시작
# 출력층에서의 오차역전파!
# 강의 참고! 출력층에서의 가중치와 바이어스의 변화율에 대한 오차역전파 공식들을 알아볼 수 있다.
# W(3) = W(3) - 알파*(E를W(3)으로 편미분) = W(3) - 알파*(A2의T승*loss_3)
# b(3) = b(3) - 알파*(E를b(3)으로 편미분) = b(3) - 알파*loss_3
# 여기서 loss_3는 1*2행렬의 ((a(3)1 - t(3)1)*a(3)1*(1-a(3)1))     ((a(3)2-t(3)2)*a(3)2*(1-a(3)2)))
# 여기서 A2는 1*2행렬은 (a(2)1   a(2)2) A2의 전치행렬은 2*1 행렬의 (a(2)1  a(2)2)

# 25강 시작
# 은닉층에서의 오차역전파!
# 강의 참고!
# 은닉층 가중치 오차역전파 공식!
# loss_2 = (loss_3 * W3전치행렬) * (A2 * (1 - A2))

# W(2) = W(2) - 알파*(E를 W(2)로 편미분)
# = W(2) - 알파*(A1전치행렬 * loss_2)

# b(2) = b(2) - 알파*(E를 b(2)로 편미분)
# = b(2) - 알파*loss_2

# 이러한 방법으로 미분없이 간단한 산술기반의 계산이 가능해짐!!

# 26강 시작
# W2, W3, b2, b3를 오차역전파로 나타내는 식을 알아냈다.
# 하지만 실제 신경망에서는 한개의 은닉층이 아닌 n개의 여러 은닉층을 사용하므로
# 그때 가중치의, 바이어스의 변화율을 구하는법을 배울 예정

# ***오차역전파 공식 구하는 순서 (은닉층이 1개인 단순 신경망으로부터 공식 패턴 추출)

# 1. 출력층의 출력 값과 정답(Target)을 이용하여, 출력층 손실 계산
# 출력층의 손실 = (출력층 출력 - 정답) * (출력층 출력(1-출력층 출력))

# 2. 은닉층 에서의 (가상의)손실 loss_3, loss_2 등을 계산할 경우, 현재 층(current layer)/
# 이전 층 (previous layer) / 다음층(next layer) 개념을 도입하여 동일한 패턴으로 반복 계산
# ex) 은닉층2를 현재층(current layer)로 설정한다면, 이전층(previous layer) 은닉층 1/
# 다음층(next layer)은 출력층으로 가정한다면 식은 다음과 같다
# 은닉층의 현재층 손실 = (다음층 손실 * 다음층에 적용되는 가중치W전치행렬) * 현재층출력 * (1-현재층출력)

# 3. 계산된 각 층의 출력 값과 손실을 이용하여, 다음식을 도출
# 현재층의 바이어스 변화율 E를b로 편미분한 값 = 현재층 손실
# 현재층에 적옹되는 가중치 변화율 E를 현재층W로 편미분한 값 = 이전층출력의 전치행렬 * 현재층 손실

# 출력층 손실 / 가중치 변화율 / 바이어스 변화율 - 1개의 은닉층을 가진 신경망 예시
# loss_3 = (A3 - Target) * A3 * (1-A3)
# W3 = W3 - 알파 * (A2전치행렬 * loss_3)
# b2 = b2 - 알파 * loss_3
# 은닉층의 손실(loss_2) = (loss_3 * W3전치행렬) * A2 * (1-A2)
# W2 = W2 - 알파 * (A1전치행렬 * loss_2)
# b2 = b2 - 알파 * loss_2


# 2개의 은닉층을 가진 신경망 예시
# loss_4 = (A4 - Target) * A4 * (1-A4)
# W4 = W4 - 알파 * (A3전치행렬 * loss_4)
# b4 = b4 - 알파 * loss_4
# 은닉층의 손실(loss_3) = (loss_4 * W4전치행렬) * A3 * (1-A3)
# W3 = W3 - 알파 * (A2전치행렬 * loss_3)
# b3 = b3 - 알파 * loss_3
# 은닉층의 손실(loss_2) = (loss_3 * W3전치행렬) * A2 * (1-A2)
# W2 = W2 - 알파 * (A1전치행렬 * loss_2)
# b2 = b2 - 알파 * loss_2

# 이러한 일정한 패턴을 이용해 오차역전파 공식 확장이 가능함!!


# 27강 시작
# 오차역전파 공식을 이용한 MNIST (필기체숫자) 인식 (최적화 작업)
# MNIST(Modified National Institute of Standards and Technology)는 손으로 직접
# 쓴 숫자(필기체 숫자)들로 이루어진 데이터 셋(Data Set) 이며,
# 우리가 새로운 프로그래밍 언어를 배울 때 'Hello, World'를 출력하는 것처럼, MNIST는
# 딥러닝을 배울 때 반드시 거쳐야 하는 'Hello, World' 같은 존재임.
# MNIST는 0부터 9까지의 숫자 이미지로 구성되며, 60,000개의 트레이닝 데이터와 10,000개의
# 테스트 데이터로 이루어져 있음

# mnist_train.csv 파일에는 학습에 이용될 수 있도록 정답(label)이 있는 총 60,000개의
# 데이터가 존재함. 1개의 데이터는 785개의 숫자가 콤마(,)로 분리되어있는데, 정답을
# 나타내는 1개의 숫자와 실제 필기체 숫자 이미지를 나타내는 784개의 숫자로 구성되어 있음
# mnist_test.csv 파일에는 총 10,000개의 데이터가 있으며, 학습을 마친 후에 구현된 딥러닝
# 아키텍처가 얼마나 잘 동작하는지 테스트 하기 위해 사용됨. 테스트 데이터 또한 정답(label)이
# 포함된 785개의 숫자로 되어 있음

# MNIST 가져오기 (numpy.loadtxt 활용)
import numpy as np
training_data = np.loadtxt('mnist_train.csv', delimiter = ',', dtype=np.float32)
test_data = np.loadtxt('mnist_test.csv', delimiter = ',', dtype=np.float32)
print("training_data.shape = ", training_data.shape, " , test_data.shape = ",test_data.shape)

# loadtxt()를 이용해서 mnist_train.csv로 부터 60,000개의 training data 와 mnist_test.csv에서
# 10,000 개의 test data를 2차원 행렬(matrix) 데이터 타입으로 가져옴

# training_data 행렬중 1개의 행(column)을 레코드라고 부름
# 1개의 레코드는 785개의 열(column)로 구성되어 있다.
# 1 열(column)에는 정답이 있음
# 2 열(column)부터 마지막 열까지는 정답을 나타내는 이미지의 색(color)을 나타내는
# 숫자 값들이 784개 연속으로 있음.

# 흑백 이미지 표현 할때 숫자 0에 가까울 수록 검은색으로, 255에 가까울수록 하얀색으로
# 나타내는데 2열부터 마지막 열까지 나열된 숫자가 바로 이미지 색을 나타내는 정보임

import matplotlib.pyplot as plt

img = training_data[0][1:].reshape(28,28)

plt.imshow(img, cmap='gray')
plt.show()

# 딥러닝 아키텍처 (one-hot encoding)
# 입력층 노드(node)를 입력 데이터의 개수와 일치하도록 784개 설정.
# 즉 training_data 행렬에서 1개의 레코드(1개의 행, row)는 785개의 숫자를 포함하지만,
# 정답을 나타내는 1열을 제외하면 2열부터 785열까지 총 784개의 데이터가 숫자 이미지를
# 나타내므로 노드 개수도 입력데이터와 일치하도록 784개 설정

# 은닉층 노드(node)를 몇개로 설정한 것인가는 정해징 규칙이 없으므로 임의로 100개 설정.

# 출력층 노드(node)는 10개 설정
# 즉, 정답은 0~9 중 하나의 숫자이므로 10개의 원소를 갖는 리스트를 만들고, 리스트에서
# 가장 큰 값을 가지는 인덱스(index)를 정답으로 판단할 수 있도록 출력 노드를 10개로 설정함
# => one-hot encoding

# one-hot encoding이란?
# | 0.01 | 0.01 | 0.01 | 0.01 | 0.99 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
# 예제) 10개의 출력 노드값이 다음과 같은 경우, 인덱스가 5인 5번째 노드(y)
# 출력 값이 0.99로 가장 크기 때문에 인덱스 값 5를 정답 5로 판단함

# MNIST(필기체 숫자 인식) - 오차역전파 버전
# 1. 입력층에 데이터입력
# 2. feed forward
# 3. 손실함수값이 최소값인지 판단
# 4. 최소면 학습종료, 아니면 W2, b2, W3, b3 업데이트 시켜주고 1번부터 반복
# loss_3 = (A3 - Target) * A3 * (1 - A3)
# W3 = W3 - 알파 * (A2전치행렬 * loss_3)
# b3 = b3 - 알파 * loss_3
# loss_2 = (loss_3 * W3전치행렬) * A2 * (1 - A2)
# W2 = W2 - 알파 * (A1전치행렬 * loss_2)
# b2 = b2 - 알파 * loss_2
# 알파는 학습률

# 구현할 코드는 external function, NeuralNetwork class, usage 세부분으로 나눈다.

# external fucntion
def sigmoid(x):
    return 1/(1+np.exp(-x))

# NeuralNetwork class
from datetime import datetime # datetime.now()를 이용하여 학습 경과 시간 측정

class NeuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate): # 가중치/바이어스/각 층 출력값/학습율 초기화
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 은닉층 가중치 W2 = (784 * 100) Xavier/He 방법으로 self.W2 가중치 초기화
        self.W2 = np.random.randn(self.input_nodes, self.hidden_nodes) / np.sqrt(self.input_nodes/2)
        self.b2 = np.random.rand(self.hidden_nodes)

        # 출력층 가중치는 W3 = (100 * 10) Xavier/He 방법으로 self.W3 가중치 초기화
        self.W3 = np.random.randn(self.hidden_nodes, self.output_nodes) / np.sqrt(self.hidden_nodes/2)
        self.b3 = np.random.rand(self.output_nodes)

        # 출력층 선형회귀 값 Z3, 출력값 A3 정의 (모두 행렬로 표시)
        self.Z3 = np.zeros([1,output_nodes])
        self.A3 = np.zeros([1,output_nodes])

        # 은닉층 선형회귀 값 Z2, 출력값 A2 정의 (모두 행렬로 표시)
        self.Z2 = np.zeros([1,hidden_nodes])
        self.A2 = np.zeros([1,hidden_nodes])

        # 입력층 선형회귀 값 Z2, 출력값 A2 정의 (모두 행렬로 표시)
        self.Z1 = np.zeros([1,input_nodes])
        self.A1 = np.zeros([1,input_nodes])

        # 학습률 learning rate 초기화
        self.learning_rate = learning_rate

    def feed_forward(self): # feed forward 이용하여 손실 함수 값 계산
        delta = 1e-7 # log 무한대 발산 방지
        
        # 입력층 선형회귀 값 Z1, 출력값 A1 계산
        self.Z1 = self.input_data
        self.A1 = self.input_data

        # 은닉층 선형회귀 값 Z2, 출력값 A2 계산
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)

        # 출력층 선형회귀 값 Z3, 출력값 A3 계산
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)

        return -np.sum(self.target_data * np.log(self.A3 + delta) + (1-self.target_data) * np.log((1-self.A3)+delta))

    def loss_val(self): # 손실 함수 값 계산 (외부출력용)
        delta = 1e-7 # log 무한대 발산 방지
        
        # 입력층 선형회귀 값 Z1, 출력값 A1 계산
        self.Z1 = self.input_data
        self.A1 = self.input_data

        # 은닉층 선형회귀 값 Z2, 출력값 A2 계산
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)

        # 출력층 선형회귀 값 Z3, 출력값 A3 계산
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)

        return -np.sum(self.target_data * np.log(self.A3 + delta) + (1-self.target_data) * np.log((1-self.A3)+delta))

    def train(self, input_data, target_data): # 오차역전파 공식을 이용하여 가중치/바이어스 업데이트 input_data : 784개, target_data : 10개
        self.target_data = target_data
        self.input_data = input_data

        # 먼저 feed forward를 통해서 최종 출력값과 이를 바탕으로 현재의 에러 값 계산
        #loss_val = self.feed_forward()

        # 출력층 loss 인 loss_3 구함
        loss_3 = (self.A3 - self.target_data) * self.A3 * (1 - self.A3)
        
        # 출력층 가중치 W3, 출력층 바이어스 b3 업데이트
        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)
        self.b3 = self.b3 - self.learning_rate * loss_3

        # 은닉층 loss 인 loss_2 구함
        loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1 - self.A2)
        
        # 은닉층 가중치 W2, 은닉층 바이어스 b2 업데이트
        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)
        self.b2 = self.b2 - self.learning_rate * loss_2

    def predict(self, input_data): # 입력 데이터에 대해 미래 값 예측 input_data 는 행렬로 입력됨 즉, (1, 784) shape를 가짐
        Z2 = np.dot(input_data, self.W2) + self.b2
        A2 = sigmoid(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = sigmoid(Z3)

        predicted_num = np.argmax(A3)

        return predicted_num

    def accuracy(self,test_data): # 신경망 기반의 딥런이 아키텍처 정확도 측정 MNIST test_data는 (10,000 * 785)
        matched_list = []
        not_matched_list = []

        for index in range(len(test_data)):
            label = int(test_data[index, 0]) # test_data의 1열에 있는 정답 분리

            # one-hot encoding을 위한 데이터 정규화 (data normalize)
            data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01
            #입력데이터는 0~255 이기 때문에, 가끔 overflow 발생 따라서 784개의 모든 입력값을 0~1 사이의 값으로 normalize 함
            
            # predict 를 위해서 vector 을 matrix 로 변환하여 인수로 넘겨줌
            predict_num = self.predict(np.array(data, ndmin=2))

            if label == predict_num:
                matched_list.append(index) # 정답과 예측 값이 맞으면 matched_list 에 추가
            else:
                not_matched_list.append(index) # 정답과 예측 값이 틀리면 not_matched_list 에 추가

        print("Current Accuracy = ", 100*(len(matched_list)/(len(test_data)))," %") # 정확도 계산 (정답데이터/전체 테스트데이터)
        return matched_list, not_matched_list

# usage

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epochs = 1
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate) # 입력노드 784개, 은닉노드 100개, 출력노드 10개, 학습률0.3 객체 생성

start_time = datetime.now()

for i in range(epochs): # 학습 반복 횟수

    for step in range(len(training_data)): # 60,000 개의 training data 를 이용하여 학습 진행
        
        # one-hot encoding 을 위한 데이터 정규화 작업 수행
        # input_data, target_data normalize
        target_data = np.zeros(output_nodes) + 0.01
        target_data[int(training_data[step, 0])] = 0.99

        input_data = ((training_data[step,1:]/255.0) * 0.99) + 0.01

        nn.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2)) # training data 를 이용하여 학습 진행

        if step % 400 == 0:
            print("step =",step, ", loss_val = ",nn.loss_val())

end_time = datetime.now()
print("\nelpsed time =", end_time - start_time)

nn.accuracy(test_data)
