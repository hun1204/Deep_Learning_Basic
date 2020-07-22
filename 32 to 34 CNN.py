# 32강 시작!
# CNN에서 사용되는 CNN연산, 풀링, 패딩 연산에 대해 알아볼 예정
# CNN과 NN의 동작원리는 같다. 다른점은 은닉층 대신 컨볼루션층을 사용하며
# 여러개의 컨볼루션층과 출력층 앞에 완전연결층이 존재한다.
# NN의 은닉층 -> 여러개의 컨볼루션층, 완전연결층

# conv / pooling
# conv(컨볼루션, convolution)
# conv는 데이터의 특징을 추출해내는 역할
# 입력데이터(A1, A2, ...)와 가중치들의 집합체인 다양한 필터(filter)와의 컨볼루션 연산을 통해
# 입력데이터의 특징(Feature)을 추출하는 역할을 수행함
# ex) A1 * filter_1 + b2 = 입력데이터 A1 특징(feature) 추출
# A2 * filter_2 + b3 = 입력데이터 A2 특징 (feature) 추출
# relu 값이 0보다 작으면 0으로 출력 0보다크면 있는 그대로의 값 출력
# pooling(풀링)
# 입력 정보를 최대값, 최소값, 평균값 등으로 압축하여 데이터 연산량을 줄여주는 역할 수행
# max pooling, min pooling, average pooling을 이용해 여러 데이터을 하나의 대표값으로 넘겨준다. 보통 max를 많이 씀

# 컨볼루션 (convolution)연산 '*' - 특징 추출 (특징 맵, feature map)
# 44행열 * 33필터 = 22컨볼루션 연산값이 나오며 + b 를통해 특징맵(feature map) 탄생
# 필터를 일정간격(스트라이드)으로 이동해 가면서, 입력데이터와 필터에서 대응하는 원소끼리
# 곱한 후 그 값들을 모두 더해주는 연산
# ex) (0:3,0:3)*필터 => 컨볼루션연산(0,0)에 입력
# (1:,0:3)*필터 => 컨볼루션연산(0,1)에 입력 (이 때 스트라이드(필터의 이동간격)는 1이다.)
# (0:3,1:)*필터 => 컨볼루션연산(1,0)에 입력
# (1:,1:)*필터 => 컨볼루션연산(1,1)에 입력
# 이때 컨벌루션 연산 *는 각 행열간의 같은 위치의 값들을 곱해서 전부 더해주는 역할

# relu 연산/ pooling 연산
# conv를 통해 나온 22특징맵(feature map)을 C2라고 칭하면
# C2를 relu연산을 거치면 Z2가 만들어진다. 이때 0보다 작은 행렬 값은 0으로 바뀐다.
# Z2가 max pooling연산을 거치면 가장큰 값을 취해서 11행열의 A2값이 나오게 된다. (MAXPOOLING이 가장 많이 사용됨)

# 패딩(padding)
# 패딩(padding)이란 컨볼루션 연산을 수행하기 전에 입력 데이터 주변을 특정 값(예를들면 0)
# 으로 채우는 것을 말하며, 컨볼루션 연산에서 자주 이용되는 방법
# -> 컨볼루션 연산을 수행하면 데이터 크기(shape)이 줄어드는 단점을 방지하기 위해 사용
# 44 * 33 -> 22 가 되는데 패딩을 사용하면 66 * 33 -> 44(feature map)이 만들어지게 된다.
# 패딩을 사용하면 아무리 많은 컨볼루션 연산을 반복하더라도 크기가 줄어들지 않을 것이다.

# 컨볼루션 연산을 통한 출력 데이터 크기(shape) 계산
# 입력 데이터 크기 (H, W)행렬로 나타내고, 필터크기(FH, FW), 패딩 P, 스트라이드 S 일 때
# 출력데이터의 크기는 (OH, OW)라고 한다.
# OH = (H + 2P -FH)/S + 1, OW = (W + 2P - FW)/S + 1 로 나타낼 수 있다.
# ex1) 입력 (4,4) 필터(3,3) 패딩1, 스트라이드1 -> 출력 (4,4)
# ex2) 입력 (28,31), 필터(5,5), 패딩2, 스트라이드3 -> 출력 (10,11)

# 33강 시작!
# 컨볼루션 층에서는,
# 1. 입력데이터 1개 이상의 필터들과의 컨볼루션 연산을 통해서
# 2. 입력데이터 특징(feature)을 추출하여 특징 맵(feature map)을 만들고,
# 3. 특징 맵(feature map) 에서 최대 값을 뽑아내서 다음 층으로 전달
# 필터를 통해 데이터 특징을 추출??

# 추출하는 방법
# 입력데이터 1개 (숫자2)에 필터 3개(가로,대각선, 세로 필터) 적용 (계산편의를 위해 패딩 적용X)
# 33행렬에 값이 존재하는 곳에 1표시 나머지는 0으로 표현
# 66행렬(숫자2)에 33가로필터 스트라이드 1, 패딩X, 바이어스 -1을 적용하면
# 총 매칭되는 데이터는 16개가 발생하고 44행렬에 바이어스를 더하면 특징 맵(feature map)이 추출된다.
# feature map을 ReLU를 통해 0과 나머지 데이터들의 44행렬로 만들어주고
# 22행렬영역마다 max pooling을 실행해 22 풀링데이터를 추출한다.

# 컨볼루션 연산 결과인 특징맵(feature) 값을 압축하고 있는 풀링 값을 보면,
# 대각선 필터에 대한 풀링 값이 가로와 세로필터의 풀링 값 보다 큰 값으로 구성되어 있는데
# 풀링 값이 크다는 것은, 데이터 안에 해당 필터의 특징(성분)이 많이 포함되어 있는 것을 의미함,
# 즉, 특징 맵 값이 압축되어 있는 풀링 결과 값을 통해 데이터의 특징(성분)을 추출할 수 있음
# 예제를 보면, 입력데이터 '2'는 대각선 특징이 가로나 세로 특징보다 더욱 많이 포함되어 있으며
# 이러한 특징을 추출하는데 대각선 필터가 가로나 세로보다 유용하다는 것을 알 수 있음.

# 34강 시작!
# 99% 이상의 정확도로 MNIST를 인식하는 CNN 코드
# 아키텍처는
# 1. 데이터분리
# 2. X, T 입력
# 3. feed forward 수행 (컨볼루션층 3개 사용, 완전연결층 사용)
# 4. y 값으로 손실함수를 계산해 최소값이면 학습종료 아니면 5. 수행
# 5. update F2, F3, F4, W5, b2, b3, b4, b5 using Optimizer => repeat 2.
# conv 연산은 tf.nn.conv2d()
# relu 연산은 tf.nn.relu()
# pooling 연산은 tf.nn.max_pool()
# 완전연결층 - 컨볼루션 층의 3차원 출력 값을 1차원 벡터로 평탄화 작업 수행하여 일반신경망 연결처럼
# 출력층의 모든 노드와 연결시켜주는 역할 수행 FLATTEN 연산은 tf.reshape(A4,...)
# 출력층 - 입력받은 값을 출력으로 0~1 사이의 값으로 모두 정규화하여 출력값들의 총합은 
# 항상 1이 되도록 하는 역할 수행 softmax 연산은 tf.nn.softmax(Z5)

# Tensorflow API - tf.nn conv2d(input, filter, strides, padding, ...)
# 최소 4개의 파라미터를 가지고있음
# 1. input : 컨볼루션 연산을 위한 입력 데이터이며 [bathch, in_height, in_width, in_channels]
# 예를들어, 100개의 배치로 묶은 28*28크기의 흑백 이미지를 입력으로 넣을경우 input은 [100,28,28,1]로 나타냄
# 2. filter : 컨볼루션 연산에 적용할 필터이며 [filter_height, filter_width, in_channels, out_channels]
# 예를들어, 필터 크기 3*3이며 입력채널 개수는 1이고 적용되는 필터 개수가 총 32개이면 filter는
# [3,3,1,32]로 나타냄
# 여기서 입력채널이란? 직관적으로 데이터가 들어오는 통로라고 생각하면 무난함. 즉, 입력채널이 1이라고 하면
# 데이터가 들어오는 통로가 1개 이며, 만약 입력채널이 32라고 하면 32개의 통로를 통해서 입력데이터가 
# 들어온다고 판단하면 무리가 없음.
# 3. strides : 컨볼루션 연산을 위해 필터를 이동시키는 간격을 나타냄, 예를들어 [1,1,1,1]로 strides를 
# 나타낸다면 컨볼루션 적용을 위해 1칸씩 이동하는것을 의미함
# 4. padding : 'SAME' 또는 'VALID' 값을 가짐, padding='VALID'라면 컨볼루션 연산 공식에 의해서
# 가로/세로(차원)크기가 축소된 결과가 리턴됨, 그러나 padding='SAME'으로 지정하면 입력값의 가로/세로(차원)
# 크기와 같은 출력이 리턴되도록 작아진 차원 부분에 0 값을 채운 제로패딩을 수행함

# Tensorflow API - tf.nn.max_pool(value, ksize,strides,padding,...)
# 1. value : [batch, height, width, channels] 형식의 입력데이터. 일반적으로 relu를 통과한 출력결과를 말하며,
# 예제에서는 Z2,Z3,Z4등의 값임.
# 2. ksize : 컨볼루션 신경망에서 일반적인 ksize는 다음과 같이 [1, height, width, 1] 형태로 표시함.
# 예를들어 ksize = [1,3,3,1] 이라고 하면 3칸씩이동, 즉 9개 (3*3) 데이터 중에서 가장 큰 값을
# 찾는다는 의미임.
# 3. strides : max_pooling을 위해 윈도우를 이동시키는 간격을 나타냄, 예를들어 [1,2,2,1]로
# strides를 나타낸다면 max pooling 적용을 위해 2칸씩 이동하는 것을 의미함
# 4. padding : max pooling 에서의 padding 값은 max pooling 을 수행하기에는 데이터가 부족한 경우에 주변을
# 0으로 채워주는 역할을 함. 예를들어 max pooling 에서 풀링층으로 들어오는 입력데이터가 77이고,
# 데이터를 2개씩 묶어 최대값을 찾아내는 연산을 하기에는 입력으로 주어진 데이터가 부족한 상황임.
# (즉, 최소 88이어야 가능), 이때 padding='SAME'이면, 부족한 데이터 부분을 0등으로 채운 후에
# 데이터를 2개씩 묶어 최대값을 뽑아낼 수 있음.

# 실제 코드 작성
# 1. read_data_set()을 통해 객체형태인 mnist로 받아오고 입력데이터와 정답데이터는 MNIST_data/
# 디렉토리에 저장이 되는데, one_hot=True 옵션을 통해 정답데이터는 one-hot encoding 형태로 저장됨
# mnist 객체는 train, test, validation 3개의 데이터 셋으로 구성되어 있으며, num_examples 값을 통해
# 데이터의 개수 확인 가능함.
# 데이터는 784(28*28)개의 픽셀을 가지는 이미지와 one-hot encoding 되어 있는 label(정답)을 가지고 있음.
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime # datetime.now()함수 이용해서 학습 경과시간 측정

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

print("\n", mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

print("\ntrain image shape = ", np.shape(mnist.train.images))
print("train label shape = ", np.shape(mnist.train.labels))
print("test image shape = ", np.shape(mnist.test.images))
print("test label shape = ", np.shape(mnist.test.labels))

# 2. X, T를 저장
# 학습율, 반복횟수, 안번에 입력으로 주어지는 데이터 개수인 배치 사이즈 등의 하이퍼 파라미터 설정
# Hyper-Parameter
learning_rate = 0.001 # 학습율
epochs = 30 # 반복횟수
batch_size = 100 # 한번에 입력으로 주어지는 MNIST 개수

# 입력과 정답을 위한 placeholder 노드 정의 (X, T)
# 입력층의 출력 값 A1은 784개의 픽셀 값을 가지고 잇는 MNIST 데이터 이지만 컨볼루션 연산을
# 수행하기 위해서 28*28*1의 차원을 가지도록 reshape 함.
X = tf.compat.v1.placeholder(tf.float32, [None, 784])
T = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 입력층의 출력 값, 컨볼루션 연산을 위해 reshape 시킴
A1 = X_img = tf.reshape(X, [-1, 28, 28, 1]) # image 28 * 28 * 1 (black/white) / 앞의 -1은 X의 크기(784)에 따라 알아서 사이즈를 맞춰준다는 의미

# 3. 컨볼루션층1
# 1번째 컨볼루션 층
# 3*3 크기를 가지는 32개의 필터를 적용
F2 = tf.Variable(tf.random.normal([3,3,1,32], stddev=0.01))
# 이때 [3,3,1,32] 에서 1은 데이터가 들어오는 통로가 1개 즉 입력 A1은 1개의 통로를 통해서 들어옴
# 표준편자 0.01
b2 = tf.Variable(tf.constant(0.1, shape=[32]))

# 1번째 컨볼루션 연산을 통해 28*28*1 => 28*28*32
C2 = tf.nn.conv2d(A1, F2, strides=[1,1,1,1], padding='SAME')

# ReLU
Z2 = tf.nn.relu(C2+b2)

# 1번째 max pooling을 통해 28*28*32 -> 14*14*32
A2 = P2 = tf.nn.max_pool(Z2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 4. 컨볼루션층2
# 2번째 컨볼루션 층
# 3*3 크기를 가지는 64개의 필터를 적용
F3 = tf.Variable(tf.random.normal([3,3,32,64], stddev=0.01))
# 이때 [3,3,32,64] 에서 32은 데이터가 들어오는 통로가 32개 즉 입력 A1은 32개의 통로를 통해서 들어옴
# 표준편자 0.01
b3 = tf.Variable(tf.constant(0.1, shape=[64]))

# 2번째 컨볼루션 연산을 통해 14*14*32 => 14*14*64
C3 = tf.nn.conv2d(A2, F3, strides=[1,1,1,1], padding='SAME')

# relu
Z3 = tf.nn.relu(C3+b3)

# 2번째 max pooling을 통해 14*14*64 -> 7*7*64
A3 = P3 = tf.nn.max_pool(Z3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

# 5. 컨볼루션층3
# 3번째 컨볼루션 층
# 3*3 크기를 가지는 128개의 필터를 적용
F4 = tf.Variable(tf.random.normal([3,3,64,128], stddev=0.01))
b4 = tf.Variable(tf.constant(0.1,shape=[128]))

# 3번째 컨볼루션 연산을 통해 7*7*64 -> 7*7*128
C4 = tf.nn.conv2d(A3,F4,strides=[1,1,1,1], padding='SAME')

# relu
Z4 = tf.nn.relu(C4+b4)

# 3번째 max pooling을 통해 7*7*128 -> 4*4*128
A4 = P4 = tf.nn.max_pool(Z4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

# 6. 완전연결층
# 4*4 크기를 가진 128개의 ACTIVATION MAP을  flatten 시킴
A4_flat = P4_flat = tf.reshape(A4, [-1, 128*4*4])
# 2048개 노드를 가지도록 RESHAPE 시켜줌, 즉 완전연결층(2048개 노드)과 출력층(10개 노드)은
# 2048*10개 노드가 완전연결 되어 있음

# 7. 출력층
W5 = tf.Variable(tf.random.normal([128*4*4, 10], stddev=0.01))
b5 = tf.Variable(tf.random.normal([10]))

# 출력층 선형회귀, 값 Z5, 즉 softmax에 들어가는 입력 값
Z5 = logits = tf.matmul(A4_flat, W5) + b5 # 선형회귀 값 Z5

y = A5 = tf.nn.softmax(Z5)

# 8. 손실함수와 Optimizer
# 출력층 선형회귀 값(logits) Z5와 정답 T를 이용하여 손실함수 크로스 엔트로피 계산
loss = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=T))

# CNN에서는 계산할량이 많기 때문에 GDA대신 성능개선을 위한 AdamOptimizer사용
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(loss)

# batch_size * 10 데이터에 대해 argmax를 통해 행단위로 비교함
predicted_val = tf.equal(tf.argmax(A5, 1), tf.argmax(T, 1))
# 출력층의 계산 값 A5와 정답 T에 대해, 행 단위로 값을 비교함

accuracy = tf.reduce_mean(tf.cast(predicted_val, tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer()) # 변수 노드(tf.Variable) 초기화
    start_time = datetime.now()
    for i in range(epochs): # 30번 반복수행
        total_batch = int(mnist.train.num_examples / batch_size) # 55,000/ 100
        for step in range(total_batch):
            batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
            loss_val, _ = sess.run([loss, train], feed_dict={X: batch_x_data, T: batch_t_data})
            if step % 100 == 0:
                print('epochs = ', i, ", step = ", step, "loss_val = ", loss_val)

    end_time = datetime.now()

    print("\nelapsed time = ", end_time - start_time)

    # Accuracy 확인
    test_x_data = mnist.test.images # 10000 * 784
    test_t_data = mnist.test.labels # 10000 * 10

    accuracy_val = sess.run(accuracy, feed_dict={X : test_x_data, T: test_t_data})

    print("\nAccuracy = ", accuracy_val)
