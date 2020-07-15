# 수치미분(Numerical Derivative)
# 수치미분은 1차함수의 기울기와 y절편을 계산하며 최적화시키기 위해 반드시 필요한 개념
# 미분(derivative),편미분,체인룰
# 입력x를 현재값에서 아주조금 변화시키면 함수 f(x)는 얼마나 변하는가
# f(x)는 입력x의 미세한 변홯에 얼마나 민감하게 반응하는가 ->미분
# 상수함수는 x값이 변해도 그대로임

# 머신러닝/딥러닝에서 자주 사용되는 함수의 미분
# f(x)=상수 -> f'(x)=0
# f(x)=axn승 -> f'(x) = nax(n-1)승
# f(x)=ex승 -> f'(x)=ex승
# f(x)=ln x -> f'(x) = 1/x   (ln은 자연로그)
# f(x)=e -x승 -> f'(x) = -e -x승

# 편미분이란? 입력변후가 하나 이상인 다변수 함수에서, 미분하고자 하는 변수 하나를 제외한 나머지
# 변수들은 상수로 취급하고, 해당 변수를 미분하는 것
# 예시(각각 변수 x,y에 대하여 편미분할경우)
# f(x,y) = 2x + 3xy + y세제곱 x로편미분-> f'(x,y) = 2 + 3y
# f(x,y) = 2x + 3xy + y세제곱 y로편미분-> f'(x,y) = 3x+3y제곱
# 편미분의 실생활 활용 예시
# 체중함수가 '체중(야식,운동)' 처럼 야식/운동에 영향을 받는 2변수 함수라고 가정할경우
# 편미분을 이용하면 각 변수 변화에 따른 체중 변화량을 구할 수 있음
# 예시(각각의 변수로 편미분)
# 현재먹는 야식의 양에서 조금 변화를 줄경우 체중은 얼마나 변하는가
# 현재하고 있는 운동량에 조금 변화를 줄경우 체중은 얼마나 변하는가

# 연쇄법칙(chain rule)이란 여러 함수로 구성됨 함수로, 이러한 합성함수를 미분하려면
# '합성함수'를 구성하는 각 함수의 미분의 곱'으로 나타내는 chain rule(연쇄법칙) 이용
# 합성함수 예시 f(x) = e(3x(제곱))승 -> 함수 e(t)승, t=3x(제곱)의 조합

#수치미분 1차버전 - numerical derivative
#수치미분은 공식쓰지않고 c/ 파이썬 등을 이용하여 주어진 입력값이 미세하게 변할 때 함수값 f는
#얼마나 변하는지를 계산해주는 것을 지칭
#수치미분 구현(1차버전)
def numerical_derivative(f,x):#f는 미분하려는 함수. 외부에서 def,lamda등으로 정의됨
    delta_x = 1e-4 #lim의 x의 증감값을 뜻하는데 이론적으로는 0을 대입해야 하지만
    #프로그래밍에서는 32bits 64bits로 정해져있기 때문에 10의 -100승 같이 너무 작은 값을 사용하면 
    #반올림오차가 발생할 수 있기때문에 10의 -4승 정도로 0에 수렴하는 값으로 사용
    return (f(x+delta_x)-f(x-delta_x)) / (2*delta_x)
#수치미분의 코드 사용예시
#함수f(x) = x제곱에서 미분계수 f'(3)을 구하기
#즉 x=3에서 값이 미세하게 변할 때, 함수 f는 얼마나 변하는지 계산하라
def my_func1(x):
    return x**2
result = numerical_derivative(my_func1,3)
print("result ==", result)

#수치미분1차버전 2번예제
#함수 f(x) = 3xe(x)승을 미분한 함수를 f'(x)라할경우 f'(2)를 구하기.
#즉 x=2에서 값이 미세하게 변할 때, 함수 f는 얼마나 변하는지 계산하라!
import numpy as np

def my_func2(x):
    return 3*x*(np.exp(x))
result = numerical_derivative(my_func2,2)
print("result ==",result)

#수치미분 최종버전!!
#입력변수가 하나 이상인 다 변수 함수의 경우, 입력변수는 서로 독립적이기 때문에
#수치미분 또한 변수의 개수만큼 개별적으로 계산하여야 함
#예제 f(x,y)의 경우 각각의 수치미분을 수행해 f'(1.0,2.0) = (8.0,15.0) 처럼 됨
def numerical_derivative_final(f, x): #f는 다변수함수일경우!, x는 모든 변수를 포함하고 있는 numpy 객체(배열,행렬)
    delta_x = 1e-4
    grad = np.zeros_like(x)#계산된 수치미분 값 저장 변수
    #모든 입력변수에 대해 편미분하기 위해 iterator 획득
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx] #numpy 타입은 mutable 이므로 원래값 보관
        
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_X)

        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x) # f(x-delta_X)
        
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        x[idx] = tmp_val
        it.iternext()

    return grad
#텐서플로우 케라스에서 실제로 쓰이는 수치미분 코드들!이다.

#11강시작! 수치미분 최종버전을 활용한 예제풀이
#수치미분 debug version!!
def numerical_derivative_debug(f, x): #f는 다변수함수일경우!, x는 모든 변수를 포함하고 있는 numpy 객체(배열,행렬)
    delta_x = 1e-4
    grad = np.zeros_like(x)#계산된 수치미분 값 저장 변수
    #모든 입력변수에 대해 편미분하기 위해 iterator 획득
    print("debug 1. initial input variable =", x)
    print("debug 2. initial grad =",grad)
    print("=========================================")
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        print("debug 3. idx =", idx, ", x[idx] =",x[idx])
        tmp_val = x[idx] #numpy 타입은 mutable 이므로 원래값 보관
        
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_X)

        x[idx] = float(tmp_val) - delta_x
        fx2 = f(x) # f(x-delta_X)
        
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        print("debug 4. grad[idx] = ",grad[idx])
        print("debug 5. grad = ",grad)
        x[idx] = tmp_val
        it.iternext()

    return grad

#입력변수가 1개인 함수 f(x) = x**2 예시
def func1(input_obj):
    x=input_obj[0]
    return x**2
numerical_derivative_debug(func1, np.array([3.0]))
#입력변수가 2개인 함수 f(x,y) = 2x + 3xy + y^3
def func2(input_obj):
    x=input_obj[0]
    y=input_obj[1]
    return (2*x + 3*x*y + np.power(y,3))
input = np.array([1.0,2.0])
numerical_derivative_debug(func2,input)
#입력변수가 4개인 함수 f(w,x,y,z) = wx+xyz+3w+zy^2
def func3(input_obj):
    w = input_obj[0,0]
    x = input_obj[0,1]
    y = input_obj[1,0]
    z = input_obj[1,1]
    return (w*x + x*y*z + 3*w + z*np.power(y,2))
input = np.array([[1.0,2.0],[3.0,4.0]])
numerical_derivative_debug(func3,input)
#수치미분예제끝! 머신러닝을 위해 필요한 파이썬, 수치미분 기본과정 수강 완료
