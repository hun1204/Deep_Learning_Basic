# 12강 시작
# 인공지능 - 인간의 학습능력, 추론능력 등을 컴퓨터를 통해 구현하는 포괄적 개념
# 머신러닝은 데이터를 이용하여 데이터 특성과 패턴을 학습하여, 그 결과를 바탕으로
# 미지의 데이터에 대한 그것의 미래결과(값, 분포)를 예측
# 데이터마이닝 : 데이터간의 상관관계나 속성을 찾는것이 주 목적

# 머신러닝은 학습의 방법에 따라 지도학습, 비지도학습으로 나뉜다. (supervised, unsupervised)
# 그 중 지도학습은 어떤종류의 미래값을 예측하느냐에 따라 regression회귀, classification분류로 나뉜다.
# 비지도학습은 학습할 데이터에 정답은 없고 입력값만 있다는것이 지도학습과의 차이점이다.
# 비지도학습은 입력값 자체의 특성과 분포만에 주목하는 clustering군집화를 사용

# 공부시간(x) 9,14,21,27,32,37
# 시험성적(t) 74,81,86,90,88,92
# 위 x,t를 training data set 이라 함

# supervised learning은 입력값x와 정답 (t,label)을 포함하는 training data를 이용하여 학습하고,
# 그 학습된 결과를 바탕으로 미지의 데이터(test data)에 대해 미래 값을 예측(predict)하는 방법
# -> 이처럼 대부분의 머신러닝 문제는 지도학습에 해당된다.
# 예시1)시험공부 시간(입력)과 PASS/FAIL(정답)을 이용한 당락 여부 예측
# 예시2)집 평수(입력)와 가격 데이터(정답)를 이용하여 임의의 평수 가격 예측

# 지도학습은 학습결과를 바탕으로, 미래의 무엇을 예측하느냐에 따라
# REGRESSION, CLASSIFICATION 회귀와 분류로 구분된다.
# Regression은 Training Data를 이용하여 연속적인(숫자) 값을 예측하는 것을 말하며
# 집평수와 가격 관계, 공부시간과 시험성적등의 관계임
# Classifiaction은 Training Data를 이용해 주어진 입력값이 어떤 종류의 값인지 구별하는것을 지칭

# 비지도학습은 Training Data에 정답은 없고 입력 데이터만 있기 때문에, 입력에 대한
# 정답을 찾는 것이 아닌 입력데이터의 패턴, 특성 등을 학습을 통해 발견하는 방법을 말함
# 예시1)Clustering군집화 알고리즘을 이용한 뉴스를 주제별로 묶는 그룹핑, 백화점 상품 추천 시스템 등
# 쉽게말하면 유사한 특성들끼리 묶는 방법

# 13강 시작 linear regression
# 오차를 기반으로하는 손실함수(loss function)에 대해 알아볼 예정
# regression - training data를 이용해 데이터의 특성과 상관관계를 파악하고,
# 그 결과를 바탕으로 미지의 데이터가 주어졌을 경우, 그 결과를 연속적인 값으로 예측하는 것

# 학습(learning)의 개념
# step1) analyze training data
# 학습데이터는 x(공부시간)에 비례해서 y(시험성적)도 증가하는 경향이 있음
# 즉, 입력x와 출력y는 y = Wx + b 형태로 나타낼 수 있다.
# training data의 특성을 가장 잘 표현할 수 있는 가중치 W(weight기울기), 
# 바이어스 b(y절편)를 찾는것이 머신러능의 학습과정 learnig이라 한다.

# 오차(error), 가중치(weight) W, 바이어스(bias) b
# training data의 정답t와 직선 y=Wx+b 값의 차이인 오차(error)는,
# 오차(error) = t-y = t-Wx+b로 계산되며, 오차가 크다면 우리가 임의로 설정한 직선의 가중치와
# 바이어스 값이 잘못된 것이고, 오차가 작으면 잘된것이기 때문에 미래 값 예측도 정확할 수 있다고 예상가능
# machine learning의 regression 시스템은, 모든 데이터의 오차(error) = t-y = t-Wx+b 의 합이 최소가 되서,
# 미래값을 잘 예측할수 있는 가중치 W와 바이어스 b 값을 찾아야 한다.

# 가중치 W와 바이어스 b를 구할수 있게해주는 손실함수(loss fucntion)!!
# loss function 혹은 cost function으로 불리는 손실함수는 training data의 정답(t)와
# 입력(x)에 대한 계산 값 y의 차이를 모두 더해 수식으로 나타낸 것
# tn-yn 각각의 오차들을 모두 더해 오차값을 계산하는데 +와 -가 더해져 오차가 0이 될수 있으므로
# 손실함수에서 오차(error)를 계산할 때는 (t-y)^2 = (t-[Wx+b])^2를 사용함.
# 즉 오차는 언제나 양수이며, 제곱을 하기 때문에 정답과 계산값 차이가 크다면, 제곱에 의해
# 오차는 더 큰 값을 가지게 되어 머신러닝학습에 있어 장점을 가짐
# loss function = E(W.b) = 1/n시그마[ti-(Wxi + b)]^2 = {(t1-y1)^2 + (t2-y2)^2 + ... + (tn-yn)^2} / n 이다.

# x와 t는 training data 에서 주어지는 값이므로, 손실함수(loss function)인 E(W,b)는 결국
# W와 b에 영향을 받는 함수임. E(W,b) 값이 작다는 것은 정답(t)와 y=Wx+b에 의해 계산된
# 값의 평균오차가 작다는 의미이며 미지의 데이터x가 주어져도 확률적으로 오차가 작을것으로 추측 가능하다.
#이처럼 training data를 바탕으로 손실함수 E(W,b)가 최소값을 갖도록 (W,b)를 구하는 것이
#(linear) regression model의 최종 목적이다.

# 14강 시작! 
# 손실함수의 최소값을 찾을 수 있는 경사하강법(gradient decent algorithm)을 알아보자!
# y= Wx+b linear regression / loss fucntion = E(W,b)
# 계산을 쉽게하고 손실함수의 모양을 파악하기 위해 E(W,b)에서 b=0을 가정
# 다음과 같은 Training Data 에서, W 값에 대한 손실함수 E(w,b)계산
# x는 1,2,3 일때 t가 1,2,3임을 가정하면 W=-1 일때 E(-1,0)=18.7 E(0,0)=4.67 E(1,0)=0 이 나온다.

# 경사하강법(gradient decent algorithm)의 원리
# 1.임의의 가중치 W 선택
# 2.그 W에서의 직선의 기울기를 나타내는 미분 값을 구함
# 3.그 미분값이 작아지는 방향으로 W값 감소시켜나가면
# 4.최종적으로 기울기가 더 작아지지 않는 곳을 찾을 수 있다. 그곳이 E(W)의 최소값임을 알 수 있다
# 이처럼, W에서의 직선의 기울기인 미분값을 이용하여, 그 값이 작아지는 방향으로 진행한다.
# 기울기가 양수이면 W값을 줄이고 기울기가 음수이면 W값을 늘리면 된다. (그래프 보면 이해 빠름)

# W값 구하기! W에서의 편미분이 해당 W에서의 기울기를 나타냄
# E'(W)가 양수이면 W값을 줄이고 기울기가 음수이면 W값을 늘린다. a(알파)학습률로 조절
# W = W - aE'(W) 이때 a(알파)를 학습률(learning rate)이라고 부르며, W값의 감소 또는 증가 되는 비율을 나타낸다.
# E(W,b) = 1/n시그마[ti-yi]^2
# 손실함수 E(W,b) 최소값을 갖는 W / W = W - aE'(W) W에 대한 편미분 
# 손실함수 E(W,b) 최소값을 갖는 b / b = b - aE'(b) b에 대한 편미분 
# a(알파)는 학습률(learning rate)라고 부르며 W값의 감소, 증가 비율을 나타냄
# gradient decent algorithm은 이러한 방법이다!

# 15강 시작!
# linear regression의 코드화 / 입력변수가 1개인 혹은 다수인 문제 해결해보기
# linear regression을 짜기 위한 5단계
# 1. 슬라이싱(slicing) 또는 list comprehension 등을 이용해 입력x와 정답t를 numpy 데이터형으로 분리
# 2. 임의의직선 y = Wx+b를 정의하는데 W와 b는 numpy.random.rand()를 이용해 0과 1사이의 랜덤한 값으로 초기화
# 3. 손실함수를 정하는데 def loss func(): y = numpy.dot(X,W) + b return (numpy.sum((t-y)**2))/(len(x))
# 위 np.dot(X, W) 에서 dot는 행렬곱으로 자주 쓰이니 익혀두기
# 4. 학습률 a(알파)는 learning_rate =  1e-3 or 1e-4 or 1e-5 ...일반적으로 사용하나
# 학습속도와 데이터의 특성에 따라 적정한값을 찾아주어야한다.
# 5. 가중치 W, 바이어스 b 구하기 
# f = lambda x: loss_func(...)
# for step in range(6000) 6000은 임의값
# W -= learning_rate*numerical_derivative(f,W)
# b -= learning_rate*numerical_derivative(f,b)

# simple regression - concept
# x의 입력값이 1,2,3,4,5 t가 2,3,4,5,6일때
# X*W + b = Y를 행렬곱 형태로 계산하면 모든 데이터X에 대해 한번에 쉽게 계산이 가능하다.

# simple regression - example

# 1.학습데이터(Training data) 준비
import numpy as np
x_data = np.array([1,2,3,4,5]).reshape(5,1)
t_data = np.array([2,3,4,5,6]).reshape(5,1)
#raw_data = [[1,2],[2,3],[3,4],[4,5],[5,6]]

# 2.임의의 직선 y = W*x + b 정의(임의의 값으로 가중치 W, 바이어스 b 초기화)
W = np.random.rand(1,1)
b = np.random.rand(1)
print("W =",W,", W.shape =",W.shape,", b =",b,", b.shape = ",b.shape)

# 3.손실함수 E(W,b)정의
def loss_func(x,t):
    y = np.dot(x,W)+b #행렬곱 이용
    return (np.sum((t-y)**2))/ (len(x))

# 4.수치미분 numerical_derivative 및 utility 함수 정의
def numerical_derivative(f, x):
    delta_x = 1e-4 #0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x,flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] =float(tmp_val)+delta_x
        fx1 = f(x)

        x[idx] =float(tmp_val)-delta_x
        fx2 = f(x)
        grad[idx] = (fx1-fx2)/(2*delta_x)
        
        x[idx] = tmp_val
        it.iternext()

    return grad

# 손실함수 값 계산함수
# 입력변수 x,t : numpy type
def error_val(x,t):
    y = np.dot(x,W)+b
    return(np.sum((t-y)**2))/ (len(x))

# 학습을 마친 후 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 x : numpy type
def predict(x):
    y = np.dot(x,W)+b
    return y

# 5.학습률(learning rate)초기화 및 손실함수가 최소가 될 때까지 W, b 업데이트
learning_rate = 1e-2 #발산하는경우, 1e-3 ~ 1e-6 등으로 바꿔서 실행
f = lambda x : loss_func(x_data,t_data) #f(x) = loss_func(x_data, t_data)
print("initial error value =",error_val(x_data,t_data),", initial W =", W,", b =",b)

for step in range(8001):
    W -= learning_rate*numerical_derivative(f,W)
    b -= learning_rate*numerical_derivative(f,b)
    if(step%400==0):
        print("step = ",step,"error value = ",error_val(x_data,t_data),"W =",W,"b = ",b)

# 학습 결과 및 입력 43에 대한 미래 값 예측
print(predict(43))

# 여러변수를 가진 regression
# multi-variable regression - concept
# 변수가 x1,x2,x3 이기때문에 결과값 y는 y= x1*W1 + x2W2 + x3W3 + b
# x*W+b = Y 계산시 (25*3)DOT(3*1) + b = (25*1)행렬로 계산됨

# 1.학습데이터(Training data)준비
loaded_data = np.loadtxt('data-01-test-score.csv',delimiter=',',dtype=np.float32)
x_data = loaded_data[:,0:-1]
t_data = loaded_data[:,[-1]]
print(x_data.shape,t_data.shape)
# 2.임의의 직선 y = W1x1 + W2x2 + W3x3 + b 정의
W = np.random.rand(3,1)
b = np.random.rand(1)
print("W =",W,", W.shape = ", W.shape, ", b=",b,", b.shape", b.shape)
# 3.손실함수 E(W,b)정의
def loss_func2(x,t):
    y = np.dot(x,W)+b
    return (np.sum((t-y)**2))/(len(x))

# 4.수치미분 numerical_derivative 및 utility 함수 정의
def numerical_derivative2(f, x):
    delta_x = 1e-4 #0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x,flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] =float(tmp_val)+delta_x
        fx1 = f(x)

        x[idx] =float(tmp_val)-delta_x
        fx2 = f(x)
        grad[idx] = (fx1-fx2)/(2*delta_x)
        
        x[idx] = tmp_val
        it.iternext()

    return grad

# 손실함수 값 계산함수
# 입력변수 x,t : numpy type
def error_val2(x,t):
    y = np.dot(x,W)+b
    return (np.sum((t-y)**2))/ (len(x))
# 학습을 마친 후 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 x : numpy type
def predict2(x):
    y = np.dot(x,W)+b
    return y


learning_rate = 1e-5 # 발산하는경우, 1e-3 ~ 1e-6 등으로 바꿔서 실행
f = lambda x : loss_func2(x_data,t_data) # f(x) = loss_func(x_data, t_data)
print("initial error value =",error_val2(x_data,t_data),", initial W =", W,", b =",b)


for step in range(15001):
    W -= learning_rate*numerical_derivative2(f,W)
    b -= learning_rate*numerical_derivative2(f,b)
    if(step%400==0):
        print("step = ",step,"error value = ",error_val2(x_data,t_data),"W =",W,"b = ",b)

test_data = np.array([100,98,81])
print(predict2(test_data))
