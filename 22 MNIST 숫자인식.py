# 22강시작!
# MNIST 필기체 숫자코드 인식
# MNIST(Modified National Institute of Standards and Technology)는 손으로
# 직접 쓴 숫자(필기체 숫자)들로 이루어진 데이터 셋(DATA SET)이다.
# 우리가 새로운 프로그래밍 언어를 배울 때 'HELLO, WORLD'를 출력하는 것처럼,
# MNIST는 딥러닝을 배울 때 반드시 거쳐야하는 'HELLO, WORLD' 같은 존재임.
# MNIST는 0부터 9까지의 숫자 이미지로 구성되며, 60,000개의 트레이닝 데이터와
# 10,000개의 테스트 데이터로 이루어져 있음

# MNIST 구조

# mnist_train.csv 파일에는 학습에 이용될 수 있도록 정답(label)이 있는 총 60,000개의
# 데이터 존재. 1개의 데이터에는 785개의 숫자가 콤마(,)로 분리되어 있는데, 정답을
# 나타내는 1개의 숫자와 필기체 숫자 이미지를 나타내는 784개의 숫자로 구성되어 있음

# minist_test.csv 파일에는 총 10,000개의 데이터가 있으며, 학습을 마친 후에 구현된
# 딥러닝 아키텍처가 얼마나 잘 동작하는지 테스트하기 위해 사용됨, 테스트 데이터 또한
# 정답(label)이 포함된 785개의 숫자로 되어 있음

import numpy as np
training_data = np.loadtxt('mnist_train.csv',delimiter=',',dtype=np.float32)
test_data = np.loadtxt('mnist_test.csv',delimiter=',',dtype=np.float32)

print("training_data.shape=",training_data.shape,", test_data.shape", test_data.shape)
# loadtxt(...)를 이용하여 minist_train.csv로 부터 60,000개의 training data와
# mnist_test.csv에서 10,000개의 test data를 2차원 행렬(matrix) 데이터 타입으로 가져옴

# training_data 행렬
# 레코드(1개의 행, row)라고 한다.
# 1. 1개의 레코드는 785개의 열(column)로 구성
# 2. 1열(column)에는 정답이 있음
# 3. 2열(column)부터 마지막 열(column) 까지는 정답을 나타내는 이미지의 색(color)을
# 나타내는 숫자 값들이 784개 연속으로 있음.
# *** 흑백 이미지 표현 할 때 숫자 0에 가까울수록 검은색으로, 255에 가까울수록 하얀색으로
# 나타내는데 2열부터 마지막 열까지 나열되는 숫자가 바로 이미지 색을 나타내는 정보임

# training_data 이미지 표현
import matplotlib.pyplot as plt
img = training_data[0][1:].reshape(28,28) # 이미지로 나타내기 위해 reshape
plt.imshow(img, cmap='gray')
plt.show()

# 딥러닝 아키텍처 (one-hot encoding)
# 입력층(784개층) : 입력층 노드(node)를 입력 데이터 개수와 일치하도록 784개 설정
# 은닉층 노드(node)를 몇개로 설정할 것인가는 정해진 규칙이 없으므로 임의로 100개 설정
# 여기서 W(2) = (784*100), W(3) = (100*10)
# 출력층 노드(node)는 10개 설정./ 즉 정답은 0~9중 하나의 숫자이므로 10개의 원소를
# 갖는 리스트를 만들고, 리스트에서 가장 큰 값을 가지는 인덱스(index)를 정답으로
# 판단할 수 있도록 출력 노드를 10개로 설정함(one-hot encoding)
# 최대값을 가지는 index를 정답으로 판단하는방식을 one-hot encoding이라고 함
# |0|1|2|3|4|5|6|7|8|9| 이때 5번노드가 0.99 값으로 가장 높은수이면 정답을 5로판단함.

# NeuralNetwork class - MNIST(필기체 숫자)인식 클래스

#1.external function
def sigmoid(x):
    return 1/1+np.exp(-x)
def numerical_derivative(f,x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val)+delta_x
        fx1=f(x)
        x[idx] = float(tmp_val)-delta_x
        fx2=f(x)

        grad[idx] = (fx1-fx2)/(2*delta_x)
        x[idx] = tmp_val
        it.iternext()

    return grad

# NeuralNetwork class
class NeuralNetwork:
    
    def __init__(self,input_nodes,hidden_nodes,output_nodes): # 가중치,바이어스,학습율 초기화
        self.input_nodes = input_nodes # input_nodes = 784
        self.hidden_nodes = hidden_nodes # hidden_nodes = 100
        self.output_nodes = output_nodes # output_nodes = 10

        # 2층 hidden layer unit
        # 가중치 W, 바이어스 b 초기화
        self.W2 = np.random.rand(self.input_nodes,self.hidden_nodes) # W2 = (784x100)
        self.b2 = np.random.rand(self.hidden_nodes) # b2 = (100,)

        # 3층 outpu layer unit
        self.W3 = np.random.rand(self.hidden_nodes, self.output_nodes) # W3 = (100X10) 
        self.b3 = np.random.rand(self.output_nodes) # b3 = (10,)

        # 학습률 learning rate 초기화
        self.learning_rate = 1e-4

    # feed_forward 를 이용하여 입력층에서 부터 출력층까지 데이터를 전달하고 손실함수 값 계산
    # loss_val(self) 메서드와 동일한 코드, loss_val(self)는 외부 출력용으로 사용함    
    def feed_forward(self): # feed forward 이용하여 손실 함수 값 계산
        delta = 1e-7 # log 무한대 발산 방지
        
        z1 = np.dot(self.input_data,self.W2) + self.b2
        y1 = sigmoid(z1)

        z2 = np.dot(y1,self.W3) + self.b3
        y = sigmoid(z2)

        # cross-entropy
        return -np.sum(self.target_data*np.log(y + delta) + (1-self.target_data)*np.log((1-y)+delta))
   
    def loss_val(self): # 손실 함수 값 계산 (외부 출력용)
        delta = 1e-7 # log 무한대 발산 방지
        
        z1 = np.dot(self.input_data,self.W2) + self.b2
        y1 = sigmoid(z1)

        z2 = np.dot(y1,self.W3) + self.b3
        y = sigmoid(z2)

        # cross-entropy
        return -np.sum(self.target_data*np.log(y + delta) + (1-self.target_data)*np.log((1-y)+delta))
    
    # input_data : 784개, target_data : 10개
    def train(self, training_data): # 수치미분을 이용하여 가중치/바이어스 업데이트
        
        # normalize
        self.target_data = np.zeros(self.output_nodes) + 0.01 # one-hot encoding 을 위한 10개의 노드 0.01 초기화
        self.target_data[int(training_data[0])] = 0.99 # 및 정답을 나타내는 인덱스에 가장 큰 값인 0.99 로 초기화

        # 입력 데이터는 0~255 이기 때문에, 가끔 overflow 발생, 따라서 모든 입력값을
        # 0~1사이의 값으로 normalize함!!
        self.input_data = (training_data[1:] / 255.0 * 0.99) + 0.01
        
        f = lambda x : self.feed_forward()

        self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
        self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
        self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
        self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)

    # query, 즉 미래 값 예측 함수
    def predict(self,input_data): # 입력 데이터에 대해 미래 값 예측
        z1 = np.dot(input_data, self.W2) + self.b2
        y1 = sigmoid(z1)

        z2 = np.dot(y1,self.W3) + self.b3
        y = sigmoid(z2)

        predicted_num = np.argmax(y) # 가장 큰 값을 가지는 인덱스를 정답으로 인식함(argmax) = one-hot encoding 구현
        
        return predicted_num

    # 정확도 측정함수
    def accuracy(self, test_data): # 신경망 기반의 딥러닝 아키첵처 정확도 측정
        
        matched_list = []
        not_matched_list = []

        for index in range(len(test_data)):
            label = int(test_data[index, 0]) # test_data의 1열의 정답 분리

            #normalize
            data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01 # 입력데이터는 0~255 이기 때문에, 가끔 overflow 발생

            predict_num = self.predict(data)
            
            if label == predict_num: # 정답과 예측값이 맞으면 matched_list에 추가
                matched_list.append(index)
            else:
                not_matched_list.append(index) # 정답과 예측값이 틀리면 not_matched list에 추가

        print("Current Accuracy = ", 100*(len(matched_list)/(len(test_data))), " %") #정확도 계산 (정답데이터/ 전체 테스트데이터)

        return matched_list, not_matched_list
# usage
# 입력노드 784개, 은닉노드 100개, 출력노드 10개의 NeuralNetwork 객체 생성
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

for step in range(30001):   # 60,000개의 training data 중 50% 데이터로 학습 진행
    index = np.random.randint(0,59999) # 60,000개 데이터 가운데 random 하게 30,000 개 선택
    nn.train(training_data[index])  # random 하게 선택된 training data를 이용하여 학습 진행

    if step % 400 == 0:
        print("step =", step,", loss_val =", nn.loss_val())

# accuracy 계산
nn.accuracy(test_data)
