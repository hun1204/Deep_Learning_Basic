# 31강 tensorflow 기초 마지막 MNIST문제 해결
# NEURAL NETWORK 동작원리
# 1. 데이터분리(입력/정답)
# 2. X, T
# 3. feed foward
# 4. y
# 5. 손실함수가 최소면 끝 아니면 update W2,W3,b2,b3 using GDA => repeat

# 이 때 은닉층의 sigmoid(z2)함수 대신 relu(z2)를 사용하며
# sigmoid 대신 relu를 사용하는 이유는 backpropagation 때 hidden layer가 많아지고
# 모든 은닉층이 sigmoid를 사용하게되면 각 단계의 계산값은 0또는1일 수 밖에 없다.
# 따라서 여러 레이어를 갖고 있을 때 최종 미분값이 0에 가까워질 수 밖에 없다.
# 이를 vanishing gradient 경사도 소실이라한다.
# sigmoid 함수는 0<n<1 사이의 값만 다루므로 결국 chain rule을 이용해 계속 값을 곱해나간다고
# 했을 때 결과 값이 0에 수렴할 수 밖에 없다는 한계를 가지고 있으므로, 
# 나중에는 1보다 작아지지 않게 하기 위한 대안으로 ReLU라는 함수를 적용하게 된다.

# relu는 쉽게 말해 0보다 작은 값이 나온 경우 0을 반환하고, 0보다 큰 값이 나온 경우 
# 그 값을 그대로 반환하는 함수다. 0보다 큰 값일 경우 1을 반환하는 sigmoid와 다르다.
# 따라서 내부 hidden layer에는 ReLU를 적용하고, 마지막 output layer에서만 sigmoid 함수를
# 적용하면 이전에 비해 정확도가 훨씬 올라가게 된다.
# 출력층에서 최종 출력값을 나타내는 sigmoid(z3) 대신 softmax(z3)를 사용하는것을 선호한다.
# softmax()는 K개의 클래스를 대상으로 일반화한 것이고
# sigmoid()는 2개의 클래스를 대상으로 한것 softmax()가 일반형이다.
# softmax()출력값들을 모두 합하면 1이 됨.

# 은닉층의 변화
# python에선 sigmoid()를 쓰지만 tensoflow에선 relu()를 사용
# 둘의 차이는 sigmoid의 출력값은 0 ~ 1 사이 이지만 ReLU(렐루)는 0보다 작은 입력값이
# 들어오면 0으로 내보내고 0보다 큰값은 그대로 내보내는 특징이 있다.
# ReLU를 선호하는 이유는 vanishing gradient 즉 경사도가 사라지는 문제를 완화시킬 수 있기 때문

# 출력층의 변화
# 성능개선을 위해 softmax()를 sigmoid()대신 사용한다.
# sigmoid의 출력값은 0~1 범위이지만 softmax는 확률분포식을 가지고 있다.
# one-hot encoding으로 나타낸 sigmoid 값을 softmax를 이용해 분류하면
# 0~9 인덱스안의 값을 확률로 나타내줌 softmax 모든 인덱스의 합은 1이된다.
# ex) 인덱스 5의 값이 0.26이면 26%확률로 가장 높아서 one-hot encodin에 의해 정답을 5로 예측

# 1. 데이터 분리
# read_data_sets()를 통해 객체형태인 mnist로 받아오고 입력데이터와 정답데이터는 MNIST_data/
# 디렉토리에 저장이 되는데, one-hot=True 옵션을 통해 정답데이터는 one-hot encoding 형태로 저장됨
# mnist 객체는 train,test, validation 3개의 데이터 셋으로 구성되어 있으며, num_examples
# 값을 통해 데이터의 개수 확인 가능함
# 데이터는 784(28*28)개의 픽셀을 가지는 이미지와 one-hot encoding 되어 있는 label(정답)을 가지고 있음

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data # 라이브러리 임포트
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("\n", mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

print("\ntrain image shape = ", np.shape(mnist.train.images))
print("train label shape = ", np.shape(mnist.train.labels))
print("test image shape = ", np.shape(mnist.test.images))
print("test label shape = ", np.shape(mnist.test.labels))

# 입력노드 784개, 은닉노드 100개, 출력노드 10개, 학습율, 반복횟수(epochs), 한번에 입력으로
# 주어지는 데이터 개수인 배치 사이즈 등 설정
learning_rate = 0.1 # 학습율
epochs = 100 # 반복횟수
batch_size = 100 # 한번에 입력으로 주어지는 MNIST 개수

input_nodes = 784 # 입력노드 개수
hidden_nodes = 100 # 은닉노드 개수
output_nodes = 10 # 출력노드 개수

# 입력괴 출력을 위한 placeholder 노드 정의 (X, T)
X = tf.compat.v1.placeholder(tf.float32,[None,input_nodes])
T = tf.compat.v1.placeholder(tf.float32,[None,output_nodes])

W2 = tf.Variable(tf.random.normal([input_nodes, hidden_nodes])) # 은닉층 가중치 노드
b2 = tf.Variable(tf.random.normal([hidden_nodes])) # 은닉층 바이어스 노드

W3 = tf.Variable(tf.random.normal([hidden_nodes,output_nodes])) # 출력층 가중치 노드
b3 = tf.Variable(tf.random.normal([output_nodes])) # 출력층 바이어스 노드

Z2 = tf.matmul(X, W2) + b2 # 선형회귀 선형회귀 값 Z2
A2 = tf.nn.relu(Z2) # 은닉층 출력 값 A2, sigmoid 대신 relu 사용

# 출력층 선형회귀 값 Z3, 즉 softmax에 들어가는 입력 값
# 출력층의 선형회귀 값을 텐서플로우에서는 logits 이라고 부름!
Z3 = logits = tf.matmul(A2,W3) + b3
y = A3 = tf.nn.softmax(Z3)

# 손실함수 값 cross-entropy 값 구하기
# 출력층 선형회귀 값(logits) Z3와 정답 T를 이용하여 손실함수 크로스 엔트로피 계산
loss = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=T))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

# batch_size X 10 데이터에 대해 argmax를 통해 행단위로 비교함
# 출력층의 계산값 A3와 정답 T 에 대해, 행 기준으로 값을 비교함
predicted_val = tf.equal(tf.argmax(A3, 1), tf.argmax(T, 1))

# batch size X 10의 True, False 를 1 또는 0 으로 변환
accuracy = tf.reduce_mean(tf.cast(predicted_val, tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer()) # 변수 노드(tf.Variable) 초기화
    for i in range(epochs): # 100번 반복수행
        total_batch = int(mnist.train.num_examples/ batch_size) # 55,000 / 100 = 550번 수행
        for step in range(total_batch):
            batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
            loss_val, _ = sess.run([loss, train], feed_dict={X: batch_x_data, T: batch_t_data})
            if step % 100 == 0:
                print("step = ", step, ", loss_val = ", loss_val)

    # Accuracy 확인
    test_x_data = mnist.test.images # 1000 x 784
    test_t_data = mnist.test.labels # 1000 x 10

    accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, T: test_t_data})

    print("\nAccuracy = ", accuracy_val)
    
# 이때 Accuracy = 0.9529
# MNIST와 같은 이미지데이터에 대한 인식정확도를 99% 이상으로 높이기 위해서는 신경망 아키텍처를 CNN
# (Convolutional Neural Network) 으로 전환하는것이 필요함
