# 29강 시작 텐서플로우2
# tensorflow with Linear Regression
# Linear Regression 복습(multi- variable example)
# 1. 데이터분리 (입력/정답)
# 2. X,T
# 3. Linear Regression (X*W+b = y)
# 4. y값 나와서 E(W,b)이 최소값인지 계산
# 5. update W,b using GDA(Gradient Descent Algorithm)경사하강법 => 2.repeat
# W = W - 알파*E(W,b)W로 편미분
# b = b - 알파*E(W,b)b로 편미분

# 이러한 linear regression을 tensor flow로 구현해 볼 예정
# 필요한 노드와 연산들은??
# 1.데이터 분리
# data-01.csv 파일로부터 (25 * 4) training data를 읽어 들인 후, 입력데이터와 정답데이터 분리
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution() # 현재2.x버전이 아닌 1.x버전의 모듈을 사용하기위해 선언

loaded_data = np.loadtxt('data-01.csv', delimiter=',')

x_data = loaded_data[ : , 0:-1] # 슬라이싱으로 1~3열까지 입력데이터로 분리
t_data = loaded_data[ : , [-1]] # 마지막열은 정답데이터로 분리

print("x_data.shape = ", x_data.shape)
print("t_data.shape = ", t_data.shape)

# 2.데이터 입력
# 가중치 W, 바이어스 b 정의 후, 입력데이터와 정답데이터 선형회귀 시스템에 입력
W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]))

X = tf.compat.v1.placeholder(tf.float32, [None,3]) # 현재 25 * 3 이지만 None 지정시 차후 50 * 3 등으로 확장이 가능함
T = tf.compat.v1.placeholder(tf.float32, [None,1]) # 지금은 none 대신 25 적어줘도 문제는 없음

# 3. 선형회귀
# y = X * W + b
y = tf.matmul(X,W) + b # 현재 X, W, b 를 바탕으로 계산된 값


# 4. 손실함수 (MSE mean square error 평균제곱 오차)
# E(W,b) = 1/n 시그마(정답-결과)^2
# reduce_mean 이나 reduce_sum은 특정 차원을 제거하고 합계, 평균을 구한다.
loss = tf.reduce_mean(tf.square(y - T)) # MSE 손실함수 정의

# 5. W,b 최적화(경사하강법 GDA gradient descendt algorithm)
# W = W - 알파*E(W,b)W로편미분
# b = b - 알파*E(W,b)b로편미분
learning_rate = 1e-5

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate) # 경사하강법 알고리즘 적용되는 optimizer

train = optimizer.minimize(loss) # optimizer를 통한 손실함수 최소화

# 이처럼 TensorFlow는 다양한 optimizer를 이용하여 손실함수를 최소화 하고, 최종적으로
# W, b를 최적화 시킴

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer()) # 변수노드 W,b를 초기화
    for step in range(8001):
        # run의 첫번째 인자값들을 연산해서 변수에 저장
        # feed_dict를 통해 입력되는 데이터를 이용하여 수행되는 연산은 loss, y, train이다.
        loss_val, y_val, _ = sess.run([loss,y,train], feed_dict={X: x_data, T: t_data})

        if step % 400 == 0:
            print("step =", step, ", loss_val", loss_val)

    print("\nPrediction is ", sess.run(y, feed_dict={X : [[100,98,81]]}))


# 30강 시작!
# tensorflow with Logistic Regression
# 1. 데이터분리(입력, 정답)
# 2. X,T값 입력
# 3. Logistic Regression
# 4. y값계산 손실함수가 최소값인지 판단
# 5. 최소값이 아니면 update W,b using GDA(Gradient Descent Algorithm)
# Logistic Regression을 정리해보면 training data를 바탕으로 선형회귀와 classification을 수행하여
# 나온 손실값을 바탕으로 업데이트하는 방법

# Training Data(diabetes.csv) 당뇨병 데이터
# shape(759 * 9) 마지막 9열은 당뇨병이면 1 아니면 0을 나타냄

# tensorflow를 통한 logistic regression 구현

# 1. 데이터분리(입력, 정답)
loaded_data = np.loadtxt("diabetes.csv", delimiter=",")

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

print("loaded_data =", loaded_data.shape)
print("x_data = ", x_data.shape, ", t_data = ", t_data.shape)


# 2. X,T값 입력
X = tf.compat.v1.placeholder(tf.float32, [None, 8]) # 현재 759*8이나, None으로 지정시 1500*8 등으로 확장 가능
T = tf.compat.v1.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random.normal([8,1]))
b = tf.Variable(tf.random.normal([1]))


# 3. Logistic Regression
z = tf.matmul(X, W) + b

y = tf.sigmoid(z)

# 4. y값계산 손실함수가 최소값인지 판단
# 손실함수는 Cross-Entropy
loss = -tf.reduce_mean(T * tf.math.log(y) + (1-T) * tf.math.log(1-y))

# 5. 최소값이 아니면 update W,b using GDA(Gradient Descent Algorithm)
# 가중치 W, 바이어스 b를 최적화 하기 위해 경사하강법GDA을 적용한 OPTIMIZER정의
learning_rate = 0.01

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

# 6. 손실함수가 최소값을 가지면 학습을 종료하고 학습이 얼마나 정확하게 이루어 졌는지,
# accuracy(정확도)를 측정하여야 함.
# 먼저 계산값이 y>0.5 면 True를 리턴하고 아니면 False를 리턴함, 즉 759개의 True와
# False에 대해서 숫자 1과 0의 타입변활을 위해 tf.cast 사용하여 예측값을 나타내는
# predicted 노드를 정의함

#시그모이드 값 y shape는 759*8*8*1 = 759*1임.
predicted = tf.round(y)


# 그리고 accuracy 노드는 1 또는 0 값을 갖는 759 개의 데이터에 대해 평균을 구하는 연산
# (tf.reduce_mean)을 정의함으로서 정확도를 계산 할 수 있음.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, T), tf.float32))

# 노드/연산 실행
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer()) # 변수 노드(tf.Variable) 초기화
    for step in range(20001):
        # feed_dict 으로 입력되는 데이터를 이용해서 수행되는 연산은 loss,train
        loss_val, _ = sess.run([loss,train], feed_dict={X: x_data, T: t_data}) 
            
        if step % 500 ==0:
            print("step = ",step,", loss_val = ", loss_val)

    # Accuracy 확인
    # feed_dict 으로 입력되는 데이터를 이용해서 수행되는 연산은 y,predicted, accuracy
    y_val, predicted_val, accuracy_val = sess.run([y,predicted, accuracy], feed_dict={X: x_data, T: t_data})
    
    print("\ny_val.shape = ", y_val.shape, ", predicted_val.shape = ", predicted_val.shape)
    print("\nAccuracy = ", accuracy_val)
