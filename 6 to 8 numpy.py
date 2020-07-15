# -*- coding: utf8 -*- 
#numpy 1강
import numpy as np

A = np.array([1,2])
print ("A ==", A, ", type ==", type(A))

from numpy import exp
result = exp(1)
print("result == ",result,"type ==", type(result))
# from numpy import *
# import numpy

# result = exp(1) + log(1.7) + sqrt(2)
# print("result ==",result,", type ==", type(result))

#numpy는 왜 필요한가! vector / matrix 생성하며
#numpy는 머신러닝 코드 개발할 경우 자주 사용되는 벡터,
#행렬 등을 표현하고 연산할 때 반드시 필요한 라이브러리
A = [[1,0],[0,1]]
B = [[1,1],[1,1]]
A + B #리스트연산처리
A = np.array([[1,0],[0,1]])
B = np.array([[1,1],[1,1]])
A+B #행렬연산

#머신러닝에서 숫자,사람,동물을 인식하기 위해서는 이미지 데이터를 행렬(matrix)로 변환하는 것이 중요하다.
#행렬연산을 위해서는 numpy 사용이 필수이다.

#벡터 생성법
#머신러닝 코드 구현시, 연산을 위해 벡터와 매트릭스 등의 형상,차원을 확인하는 작업이 필요
A = np.array([1,2,3])
B = np.array([4,5,6])
print("A == ", A,"B ==", B)
print("A.shape ==",A.shape,",B.shape ==", B.shape)
print("A.ndim ==",A.ndim,"B.ndim ==",B.ndim)

print("A + B ==", A+B)
print("A - B ==", A-B)
print("A * B ==", A*B)
print("A / B ==", A/B)

#2차원 행렬(matrix) 생성
#벡터와 마찬가지로 array()메소드로 생성
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[-1,-2,-3],[-4,-5,-6]])

print("A.shape ==",A.shape,",B.shape ==", B.shape)
print("A.ndim ==",A.ndim,"B.ndim ==",B.ndim)
#dim은 dimention의 약자로 차원을 뜻한다

#형변환(reshape)
# 벡터를 matrix로 matrix를 다른 형상의 matrix로 변경하기 위해서 reshape()를 사용해 shape를 변경함
C = np.array([1,2,3])
print("C.shape ==", C.shape)
print(C)
C = C.reshape(1,3)
print("C.shape ==", C.shape)
print(C)
#7강 행렬곱, broadcast, index/slice/iterator
#행렬곱(dot product)는 np.dot(A,B) 나타내며, 행렬 A의 열 벡터와 B행렬의 행 벡터가같아야함
#reshape또는 전치행렬 transpose등을 사용해 형 변환후 행렬곱 해야함
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[-1,-2],[-3,-4],[-5,-6]])
C = np.dot(A,B)
print("A.shape ==", A.shape,", B.shape ==", B.shape)
print("C.shape ==", C.shape)
print(C)
#행렬의 원소개수가 같아야 사칙연산이 가능한 한계를 벗어난 것이 행렬곱이다.
#그렇기 때문에 머신러닝과 이미지 프로세싱 분야에서 자주 사용됨
#예시 64*10이 필요하면 64*64*64*125*125*16*16*10 = 64*10 이된다
#행렬곱을 사용하지 않고 산술 연산만 가능하다면 입력 행렬의 64*64 크기의 값만 사용해야하기 때문에 다양한 특성을지닌 필터 개발이 불가하다.
#다양한 특성을지닌 필터 사용을 위해 행렬곱을 사용!
#행렬의 사칙연산은 두개의 크기가 같아야하는데 numpy에서는 두 행렬간에도 사칙연산이 가능하도록 해주는데
#이것을 브로드캐스트라고 지징한다 broadcast기능예시!
A = np.array([[1,2],[3,4]])
b = 5 
print(A+b) 
#이를 동일한 크기의 ([[5,5],[5,5]])로 만들어 계산해줌(이것이 broadcast)
C = np.array([[1,2],[3,4]])
D = np.array([4,5])
print(C+D) 
#[4,5]의 스칼라를 [[4,5],[4,5]]로 만들어줌
#행렬곱연산에도 broadcast 적용되지 않음 오직 사칙연산에서만 동작

#numpy 전치행렬(transpose) 예시
#원본행렬의 열을 행으로 행을 열로 바꾼것이다.
A = np.array([[1,2],[3,4],[5,6]])
B = A.T #전치행렬 3x2에서 2x3으로 바뀜
print("A.shape ==", A.shape, "B.shape ==", B.shape)
print(A)
print(B)

#벡터는 MATRIX가 아니므로 형변환을 시켜줘야 전치행렬 연산이 가능하다.
C=np.array([1,2,3,4,5])
D=C.T
E=C.reshape(1,5)
F=E.T
print("C.shape ==", C.shape, "D.shape ==", D.shape)
print("E.shape ==", E.shape, "F.shape ==", F.shape)
print(F)

#행렬의 원소값을 얻기 위해서 LIST에서 처럼 인덱싱과 슬라이싱을 사용
A = np.array([10,20,30,40,50,60]).reshape(3,2)
print("A.shape ==",A.shape)
print(A)
print("A[0,0] ==",A[0,0],", A[0][0] ==", A[0][0])
print("A[0:-1, 1:2] ==",A[0:-1, 1:2] )
print("A[:,0] ==", A[:,0])
print("A[:,:] ==",A[:,:])

#iterator기능! 행렬의 모든 원소를 처음부터 끝까지 접근하는 경우 사용
A = np.array([[10,20,30,40],[50,60,70,80]])
print(A, "\n")
print("A.shape ==", A.shape,"\n")
#행렬A의 iterator 생성
it = np.nditer(A, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    idx = it.multi_index
    print("current value =>",A[idx])
    it.iternext()

#8강 concatenate / function(loadtxt(),rand(),argmax()...)
#행렬에 행과 열을 추가하기 위해 사용되는 concatenate 함수
#***머신러닝의 회귀(regression 코드 구현시 가중치weight와 바이어스bias를 별도로 구분하지 않고 하나의 행렬로 취급하기 위한 프로그램이 구현 기술)
A = np.array([[10,20,30],[40,50,60]])
print(A.shape)
#A matrix에 행 추가할 행렬, 1행3열로 reshape
#행을 추가하기 대문에 우선 열을 3열로 만들어야 함
row_add = np.array([70,80,90]).reshape(1,3)
#열을 추가할때는 행을 2로 만들어야 함
column_add = np.array([1000,2000]).reshape(2,1)
print(column_add.shape)
#numpy.concatenate에서 axis = 0 (행 기준 옵션)
#A 행렬에 row_add 행렬 추가
B = np.concatenate((A, row_add), axis=0)
print(B)
#numpy.concatenate에서 axis = 1 (열 기준 옵션)
#A 행렬에 column_add 행렬 추가
C = np.concatenate((A, column_add), axis=1)
print(C)

#numpy의 loadtxt()함수는 파일에서 데이터를읽기위한 함수
loaded_data = np.loadtxt('data-01.csv',delimiter=',', dtype=np.float32)
x_data = loaded_data[:,0:-1]
t_data = loaded_data[:,[-1]]
# 데이터 차원 및 shape 확인
print("x_data.ndim = ", x_data.ndim,", x_data.shape = ", x_data.shape)
print("t_data.ndim = ", t_data.ndim,", t_data.shape = ", t_data.shape)
#입력데이터와 정답데이터를 구분할때 슬라이싱이 필요하다.

#rand함수와 sum,exp,log함수
random_number1 = np.random.rand(3)
random_number2 = np.random.rand(1,3)
random_number3 = np.random.rand(3,1)
print("random_number1 ==",random_number1,"random_number1.shape ==", random_number1.shape)
print("random_number2 ==",random_number2,"random_number2.shape ==", random_number2.shape)
print("random_number1 ==",random_number3,"random_number3.shape ==", random_number3.shape)
#반복문 없이도 각 원소에 대해 알아서 계산해줌
X =np.array([2,4,6,8])
print("np.sum(X) ==", np.sum(X))
print("np.exp(X) ==", np.exp(X))
print("np.log(X) ==", np.log(X))
#max,min,argmax,argmin원소의최대값 최소값 찾아주는 함수와 최대값 최소값의 인덱스 리턴해줌
X =np.array([2,4,6,8])
print("np.max(X) ==", np.max(X))
print("np.min(X) ==", np.min(X))
print("np.argmax(X) ==", np.argmax(X))
print("np.argmin(X) ==", np.argmin(X))
X = np.array([[2,4,6],[1,2,3],[0,5,8]])
print("np.max(X) ==",np.max(X, axis=0)) #axis=0 열기준
print("np.min(X) ==",np.min(X, axis=0))
print("np.max(X) ==",np.max(X, axis=1)) #axis=1 행기준
print("np.min(X) ==",np.min(X, axis=1))
print("np.argmax(X) ==",np.argmax(X, axis=0)) #axis=0 열기준
print("np.argmin(X) ==",np.argmin(X, axis=0))
print("np.argmax(X) ==",np.argmax(X, axis=1)) #axis=1 행기준
print("np.argmin(X) ==",np.argmin(X, axis=1))
#ones,zeros 1이나 0이 채워진상태의 행렬을 생성
A = np.ones([3,3])
print("A.shape ==", A.shape,", A==", A)
B = np.zeros([3,2])
print("B.shape ==", B.shape,", B==", B)

#데이터의 특성과 분포를 알아보기위해선 데이터 시각회 필요
#기초가 되는 matplotlib 일반적으로 line plot, scatter plot 많이 이용함
import matplotlib.pyplot as plt
x_data = np.random.rand(100)
y_data = np.random.rand(100)
plt.title('scatter plot')
plt.grid() #그리드 여부
#x범위,y범위,컬러,표현종류
plt.scatter(x_data, y_data, color='b', marker='o') #그래프의 종류
plt.show()

x_data = [x for x in range(-5,5)]
y_data = [y*y for y in range(-5,5)]
plt.title('line plot')
plt.grid()
plt.plot(x_data, y_data, color='b') #그래프종류 lineplot
plt.show()

x_data = [-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
y_data = [-8,-13,0,3,6,-1,-5,-7,1,8,7,12,13]
plt.title('line plot')
plt.grid(False)
plt.plot(x_data,y_data,color='b')
plt.show()
