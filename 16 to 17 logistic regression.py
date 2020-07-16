# # 16강 시작!
# # classicfication으로 불리는 logistic regression 개념과 손실함수인 cross entropy에 대한 강의
# # 분류(classification) 이란
# # training data의 특성과 관계들을 파악한 후에, 미지의 입력 데이터에 대하여
# # 결과가 어떤 종류의 값으로 분류 될 수 있는지를 예측하는 것
# # 예시) 스팸문자분류(sapam 1, ham 0), 암 판별 (악성종양 1, 종양 0) 등 여러곳에 쓰임

# # classification 방법
# # 1.input 2.learning 3.ask 4.predict 순으로 전개
# # 1. trainig data 특성과 분포를 나타내는 최적의 직선을 찾고(linear regression)
# # 2. 직선을 기준으로 데이터를 위(1) 또는 아래(0)으로 분류해주는 알고리즘
# # => 이러한 logistic regression은 classification 알고리즘 중에서도 정확도가 높은
# # 알고리즘으로 알려져 있어서 Deep learning에서 기본 componet(요소)로 사용되고 있음

# # 출력값 y가 1 또는 0만 가져야하는 분류시스템에서 sigmoid함수 사용
# # (x,t) -> regression(z = Wx + b) -> Classificaiton(sigmoid) -> y=sigmoid(z) -> y(1 or 0)
# # 즉 linear regression 출력 Wx+b 가 어떤 값을 갖더라도, 출력함수로 sigoid를 사용해서
# # sigmoid 계산 값이 0.5보다 크면 결과로 1이 나올확률이 높다는 것이니 y를 1로
# # 반대로 계산 값이 0.5 미만이면 결과로 0이 나올 확률이 높다는 것이니 y를 0로 정의해 classification 구현함
# # 다르게보면 sigmoid는 결과가 나타날 확률을 의미함!

# # 손실함수(loss fucntion) cross-entropy 유도 (어려워서 스킵해도 된다함)

# # classification 최종 출력 값 y는 sigmoid 함수에 의해 0~1 사이의 값을 갖는 확률적인
# # 분류모델이므로, 다음과 같이 확률변수 C를 이용해 출력 값을 나타낼 수 있다. 
# # 1. p(C=1|x) = y = sigmoid(Wx+b) 입력 x에 대해 출력 값이 1일 확률을 y로 정의.
# # 2. p(C=0|x) = 1 - p(C=1|x) = 1-y 입력 x에 대해 출력값이 0일 확률이며, 확률은 모두 더한것이
# # 1이므로 출력값이 0일 확률을 1-y임
# # 3. 위의 식을 일반형으로 나타내면 p(C=t|x) = y^t * (1-y)^(1-t) 
# # 확률변수 C는 0이나 1 밖에 가질수 없으므로 다음식과같이 나타낼 수 있는 것이다.(정답은 t=0 or 1)
# # 우리가 취급하는 training data 는 다수의 x값들을 입력하므로 전체입력값 x에 대하여
# # 최종 출력값이 0이나 1이 가장 많이 나올 확률을 구해야함
# # 4. -> L(W,b) 우도함수. 다수입력 x에 대해 정답 t가 발생될 확률을 나타낸 함수
# # 우도함수는 모든 입력값에 대해 발생확률을 나타내야하기 때문에 파이를 이용해 확률변수를 곱해줌
# # 확률은 독립적이므로, 각 입력 데이터의 발생 확률을 곱해서 우도함수를 나타낸다.
# # 파이y^t * (1-y)^(1-t)
# # 우도함수는 발생확률을 나타내는 것이기 때문에 우도함수가 최대라는것은 발생확률이 가장 높다는 것이다.
# # 모든 입력데이터에 대하여 발생확률을 곱한것(우도함수)이므로 각 입력에 대하여 나타날 확률도 가장 높다는 것을 의미
# # L(W,b)가 최대가 되도록 즉 최종출력값인 1 또는 0이 가장많이 발생할 확률이 최대가 되도록
# # 가중치와 바이어스의 편미분을 통해 업데이트를 해야하는데 우도함수 자체가 곱하기여서 미분이 어렵다.
# # 미분을 편히 하기위해 log를 취해 덧셈형태로 바꾸어주고, 함수에 들어있는 W와 b를 편미분 해준다.
# # E(W,b) = -logL(W.b) 로 정의하면 최대값의 문제가 최소화 문제로 바뀌므로 계산가능
# # 최종목적은 W와 b구하기!!!

# # Classificaiton에서의 [W,b] 계산 프로세스 정리!!
# # 1. trainig data를 입력받아 임의의 직선 Wx+b linear regression
# # 2. 결과값 Wx+b를 Classificaiton sigmoid함수의 입력값으로 사용 결과값 0 또는 1 생성
# # 3. cross entropy를 이용해 E(W,b) 손실함수를 계산한다.
# # 4. 손실함수가 최소값인지 판단하여 맞으면 학습을 종료하고 아니면 수치미분을 이용해
# # 가중치 W와 b를 업데이트하고 다시 1,2,3,4반복함

# # 17강 시작!
# # logistic regression 코드로 나타내는 방법
# # 1. slicing 또는 list comprehension을 이용해 입력값 x와 정답 t를 numpy 타입으로 분리(이때 t는 0 또는 1)
# # 2. z=Wx+b구현 W = numpy.random.rand(...)..., b = numpy.random().rand(...)...
# # 3. 0~1사이의 값을 갖는 시그모이드 함수 구현 -> def sigmoid(x): return 1/ (1+numpy.exp(-x))
# # z = Wx+b, y = sigmoid(z)를 나타낸 것. 이제 손실함수(크로스 엔트로피)를 구현
# # def loss_func(...): delta = 1e-7    z=numpy.dot(X.W)+b     y=sigmoid(z)
# # return -numpy.sum(t*numpy.log(y+delta)+(1-t)*numpy.log(1-y+delta)) #이 때 로그의 무한대를 방지하기위해 delta더해줌
# # 4. 학습률 알파값을 정의 learning_rate = 1e-3 or 1e-4 or 1e-5 ... 임의로 지정
# # 5. 가중치 W와 바이어스 b를 구해야한다. f = lambda x : loss_func(...)
# # for step in rang(6000): W -=learning_rate*numerical derivative    b-= learning_rate*numerical_derivative(f,b)

# # 입력변수 x가 1개인 logistic regression 예시
# # training data 공부시간 x 합/불합 t 로 정의 후 위 5가지 방법으로 순서대로 계산

# # 코드예시
# # 1. 학습데이터(Training Data) 준비
# import numpy as np
# x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(10,1)
# t_data = np.array([0,0,0,0,0,0,1,1,1,1]).reshape(10,1)
# # 2. 임의의 직선 z = W*x+b정의(임의의값으로 가중치W, 바이어스 b 초기화)
# W = np.random.rand(1,1)
# b = np.random.rand(1)
# print("W = ",W,"W.shape = ",W.shape,", b = ",b,", b.shape = ", b.shape)
# # 3. 손실함수 E(W,b)정의
# def sigmoid(x):
#     return 1 / (1+np.exp(-x))
# def loss_func(x,t):
#     delta = 1e-7 #무한대 발산 방지
#     z = np.dot(x,W)+b
#     y = sigmoid(z)
#     #cross-entropy sum(t-y)^2를 다음과같이 나타낸다!
#     return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))
# # 4. tnclalqns numerical_derivative 및 utility 함수 정의
# def numerical_derivative(f,x):
#     delta_x = 1e-4
#     grad = np.zeros_like(x)

#     it = np.nditer(x,flags=['multi_index'], op_flags=['readwrite'])

#     while not it.finished:
#         idx = it.multi_index
#         tmp_val = x[idx]
#         x[idx] = float(tmp_val) + delta_x
#         fx1 = f(x)
#         x[idx] = float(tmp_val) - delta_x
#         fx2 = f(x)
#         grad[idx] = (fx1-fx2)/(2*delta_x)

#         x[idx] = tmp_val
#         it.iternext()
    
#     return grad

# def error_val(x,t):
#     delta = 1e-7
    
#     z = np.dot(x,W) + b
#     y = sigmoid(z)

#     #cross-entropy
#     return -np.sum(t*np.log(y-delta)+ (1-t)*np.log((1-y)+delta))

# def predict(x):
#     z = np.dot(x,W) + b
#     y = sigmoid(z)

#     if y>0.5:
#         result = 1
#     else:
#         result = 0
    
#     return y,result

# # 5. 학습율(learning rate) 초기화 및 손실함수 최소가 될 때까지 W, b 업데이트
# learning_rate = 1e-2 #발산하는경우, 1e-3~ 1e-6 등으로 바꾸어서 실행
# f = lambda x : loss_func(x_data,t_data)

# print("initial error value = ", error_val(x_data,t_data), "initial W =", W, "\n",", b = ", b)

# for step in range(10001):
#     W -= learning_rate*numerical_derivative(f, W)
#     b -= learning_rate*numerical_derivative(f, b)
#     if(step % 400 == 0):
#         print("step = ",step, "error value = ",error_val(x_data,t_data),"W = ", W,", b =",b)
    
# (rea_val, logical_val) = predict(3)
# print(rea_val,logical_val)
# (rea_val, logical_val) = predict(17)
# print(rea_val,logical_val)




# 입력변수가 여러개인 logistic regression 예제코드!
# 입력데이터가 2개인 경우! 가중치 W를 2*1 행렬로 곱해줘야 함
# 1. 학습데이터(Training Data) 준비
import numpy as np
x_data = np.array([[2,4],[4,11],[6,6],[8,5],[10,7],[12,16],[14,8],[16,3],[18,7]])
t_data = np.array([0,0,0,0,1,1,1,1,1]).reshape(9,1)
# 2. 임의의 직선 z = w1x1 + w2x2 + b 정의 (가중치 W, 바이어스 b 초기화)
W = np.random.rand(2,1)
b = np.random.rand(1)
print("W =", W,", W.shape = ", W.shape,", b = ",b,", b.shape = ", b.shape)
# 3. 손실함수 E(W,b) 정의
def sigmoid(x):
    return 1 / (1+np.exp(-x))
def loss_func(x,t):
    delta = 1e-7 # log 무한대 발산 방지
    z = np.dot(x,W) + b
    y = sigmoid(z)
    #cross-entropy를 통해 sum(t-y)^2를 다음과같이 나타냄
    return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))
# 4. 수치미분 numerical_derivative 및 utility 함수 정의
def numerical_derivative(f,x):
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'],op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val)+delta_x
        fx1 = f(x) # f(x+delta_x)

        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad

def error_val(x,t):
    delta = 1e-7 # log 무한대 발산 방지
    z = np.dot(x,W) + b
    y = sigmoid(z)
    # cross-entropy sum(t-y)^2을 나타낸 것
    return -np.sum(t*np.log(y+delta)+ (1-t)*np.log((1-y)+delta))

def predict(x):
    z= np.dot(x,W)+b
    y= sigmoid(z)

    if y > 0.5:
        result = 1 #true
    else:
        result = 0 #false

    return y, result

# 5. 학습율(learning rate)초기화 및 손실함수가 최소가 될 때까지 W,b 업데이트

learning_rate = 1e-2 # 1e-2,1e-3은 손실함수 값 발산
f = lambda x : loss_func(x_data,t_data)

print("initial error value = ", error_val(x_data,t_data), "initial W =", W, "\n",", b = ", b)

for step in range(80001):
    W -= learning_rate*numerical_derivative(f, W)
    b -= learning_rate*numerical_derivative(f, b)
    
    if(step % 400 == 0):
        print("step = ",step, "error value = ",error_val(x_data,t_data),"W = ", W,", b =",b, "w[0]/w[1] = ",float(W[0,0]/W[1,0]))

test_data = np.array([3,17]) #예습시간, 복습시간
print(predict(test_data))
test_data = np.array([5,8])
print(predict(test_data))
test_data = np.array([7,21])
print(predict(test_data))
test_data = np.array([12,0])
print(predict(test_data))

# 0, 0, 1, 1이 나오는데
# 미래값을 예측해보자면, 복습보다는 예습시간이 합격(Pass)에 미치는 영향이 크다는 것을 알 수 있다.
# (즉, 예습시간에 대한 가중치 W1 =2.28, 복습시간에 대한 가중치 W2=1.06 에서 보듯이 예습시간이
# 복습시간에 비해 최종결과에 미치는 영향이 2배 이상임)
