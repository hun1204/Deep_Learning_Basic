# 28강 시작
# TensorFlow - 설치
# TensorFlow는 Google에서 개발하고 공개한 머신러닝/딥러닝 라이브러리
# - C++,Java 등의 다양한 언어를 지원하지만 파이썬(Python)에 최적화 되어있음

# 윈도우 환경에 TensorFlow 설치
# cmd창에 pip install tensorflow 설치하면 됨

# TensorFlow는 이름이 나타내고 있는 것처럼 텐서(Tensor)를 흘려보내면서(Flow)
# 머신러닝과 딥러닝 알고리즘을 수행하는 라이브러리임.
# - 숫자 1 (스칼라 또는 rank 0 텐서)
# - 1차원 배열 [1,2] (벡터 또는 rank 1 텐서)
# - 2차원 배열 [[1,2],[3,4]] (행렬 또는 rank 2 텐서)
# - 3차원 배열 [[[1,2]],[[3,4]]] (텐서 또는 rank 3 텐서)
# 이렇게 모든 데이터를 텐서로 취급이 가능하다.

# 이러한 텐서들은 그래프(Graph) 구조에서 노드(Node)에서 노드로 흘러감(Flow) - 이 때 노드는 상수, 변수, 텐서연산(+,-,행렬곱,컨벌루션연산등)을 나타냄
# - 그래프 자료구조는 노드(Node)와 엣지(Edge)로 구성됨
# - 텐서플로를 이용한 프로그램 작성시,
# 1) 상수, 변수, 텐서연산 등의 노드와 엣지를 먼저 정의하고,
# 2) 세션을 만들고 그 세션을 통해 노드간의 데이터(텐서) 연산 수행
# (node는 동그라미 Edge는 여러 node들을 이어주는 선을 의미)

# TensorFlow 상수 노드 - tf.constant(...)

import tensorflow as tf
tf.compat.v1.disable_eager_execution() # 현재2.x버전이 아닌 1.x버전의 모듈을 사용하기위해 선언
print(tf.__version__)

# 상수 노드 정의 / 상수값을 저장하는 노드를 만들기 위해서 상수노드 tf.constant 정의
# a와 b는 시각화 툴인 텐서보드에서 name 지정
a = tf.constant(1.0, name='a')
b = tf.constant(2.0, name='b')
c = tf.constant([[1.0,2.0],[3.0,4.0]])

# 세션을 만들지 않고 print와 같은 명령문을 실행하면, 저장된 값이 아닌 현재
# 정의되어 있는 노드의 상태(노드타입, shape 등)가 출력됨
print(a)
print(b)
print(c)

# 노드간의 연산을 위해 세션 생성
# 세션 (session)을 만들고 노드간의 텐서 연산 실행

sess = tf.compat.v1.Session()

# 세션을 통해 (sess.run()) 노드에 값이 할당되고 노드간의 텐서를 
# 흘려보내면서(tensor flow) 연산과 명령문 등이 실행됨
print(sess.run([a,b]))
print(sess.run(c))
print(sess.run([a+b]))
print(sess.run(c+1.0)) # broadcast 수행

# 세션 close
# 생성된 세션 close
# sess.close()

# TensorFlow 플레이스홀더 노드 - tf.placeholder(...)

# 플레이스홀더 노드 정의
# 텐서플로에서는 임의의 값을 입력으로 받기 위해 플레이스홀더 노드(tf.placeholder)정의
a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
c = a + b

# 세션 (session)을 만들고 플레이스홀더 노드를 통해 값 입력받음
# 노드간의 연산을 위해 세션 생성 
# sess = tf.compat.v1.Session()

# 플레이스홀더 노드에 실제 값을 넣어줄 때는 sess.run 첫번째 인자로 실행하고자 하는
# 연산(c)을 넣어주고, 두번째 인자에는 실제로 넣을 값들을 Dictionary 형태로 넣어주는
# feed_dict을 선언하고, feed_dict 부분에 플레이스홀더에 넣을 값을 지정해줌
print(sess.run(c, feed_dict={a: 1.0, b: 3.0}))
print(sess.run(c, feed_dict={a: [1.0, 2.0], b: [3.0, 4.0]}))

# 연산 추가
d = 100 * c

print(sess.run(d,feed_dict={a: 1.0, b: 3.0}))
print(sess.run(d,feed_dict={a: [1.0, 2.0], b: [3.0, 4.0]}))

# 세션 close
# 생성된 세션 close
# sess.close()

# 이렇게 플레이스홀더 노드는 머신러닝/딥러닝에서 입력데이터(input),
# 정답데이터(target)를 넣어주기 위한 용도로 주로 사용됨.

# TensorFlow 변수 노드 - tf.Variable(...)
# 변수를 저장하는 variable노드
# 가중치나 바이어스처럼 계속 업데이트 되는 변수 노드 (tf.Variable)로 정의

# 값이 계속 업데이트되는 변수노드 정의
W1 = tf.Variable(tf.random.normal([1])) # W1,b1 = np.random.rand(1)이랑 비슷함
b1 = tf.Variable(tf.random.normal([1]))

W2 = tf.Variable(tf.random.normal([1,2])) # W2,b2 = np.random.rand(1,2)이랑 비슷함
b2 = tf.Variable(tf.random.normal([1,2]))

# 노드간의 연산을 위해 세션 생성
# sess = tf.compat.v1.Session()

# 변수노드 값 초기화, 변수노드를 정의했다면 반드시 필요함
# 변수 초기화를 위한 global_variables_initializer()
sess.run(tf.compat.v1.global_variables_initializer())

# 변수노드 값 업데이트
for step in range(3):
    W1 = W1 - step # W1변수노드 업데이트
    b1 = b1 - step # b1변수노드 업데이트

    W2 = W2 - step # W2변수노드 업데이트
    b2 = b2 - step # b2변수노드 업데이트

    print("step = ",step,", W1 = ", sess.run(W1),", b1 = ", sess.run(b1))
    print("step = ",step,", W2 = ", sess.run(W2),", b2 = ", sess.run(b2))

# 생성된 세션 close
# 세션 close
sess.close()

# tf.Variable(...)에서 사용되는 초기값
# tf.random_normal, tf.truncated_normal, tf.random_uniform, tf.ones, tf.zeros
# tf.constant 등이 있음
