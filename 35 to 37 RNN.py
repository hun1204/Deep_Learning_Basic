# 35강 시작!
# RNN(Recurrent Neural Network)의 개념, 기본구조, 동작원리
# NN vs. RNN
# NN은 입력층, 은닉층, 출력층을 통해 입력 후 feed foward 수행 후 손실함수 계산해서
# 최소가 될때까지 optimizer를 이용해 가중치와 바이어스를 업데이트 하는 구조이다.
 
# RNN도 마찬가지로 입력, 은닉, 출력층을 가지고 있으며 feed forward와 update까지 다 같다.
# 다른점은 두가지가 있는데 NN은닉층의 출력값이 relu를 사용하는 반면 RNN은 활성화 함수로
# tanh를 사용한다는 점과 가장 중요한 부분인 순환 구조가 은닉층에 존재한다는 점이다. (tanh는 하이퍼블릭 탄젠트)
# NN에서는 활성화함수 계산 후 바로 다음층에 입력하는데 RNN은 은닉층의 출력값을 출력층에 전달해줌과
# 동시에 은닉층에서 순환되어서 입력값으로 다시 들어간다는 점이 다르다.


# RNN - 은닉층 내부에 순환구조를 가지고 있는 신경망
# 1. 내부적으로 순환(Recurrent) 되는 구조를 이용하여,
# 2. 순서(sequence)가 있는 데이터를 처리하는 데 강점을 가진 신경망
# 순서(sequence)가 있는 데이터란???

# I work at google 나는 구글에 근무한다. (work동사, google명사)
# I google at work 나는 회사에서 구글링한다. (google동사, work명사)
# 문장에서 단어는 순서에 따라 의미가 달라질 수 있다.
# 순서(sequence)가 있는 데이터란?
# 문장이나 음성과 같은 연속적인 데이터를 말하는데, 이런 데이터는 문장에서 놓여진 위치(순서)에 따라 
# 의미가 달라지는 것을 알 수 있음
# 즉, 현재 데이터 의미를 알기 위해서는 이전에 놓여 있던 과거 데이터도 알고 있어야 함
# (I work/ I google [대명사 + 동사], at google/ at work [전치사 + 명사])
# 그래서 RNN은 이러한 과거의 데이터를 알기 위해서 
# 1. 은닉층내에 순환(Recurrent) 구조를이용하여 과거의 데이터를 기억해 두고 있다가 
# 2. 새롭게 입력으로 주어지는 데이터와 은닉층에서 기억하고 있는 과거 데이터를 연결 시켜서
# 그 의미를 알아내는 기능을 가지고 있음

# 예시)
# 1. I입력 -> 은닉층 tanh 계산값 출력층 전달, 순환구조를 통해 I 데이터 기억 -> 출력층SOFTMAX -> [대명사,명사,동사,전치사]
# 2. work입력 -> work와 기억된 I를 연결시켜 tanh 계산값 출력층 전달, 순환구조를 통해 I의 영향을 받은
# work가 기억됨 -> 출력층softmax [대명사,명사,동사,전치사] 대명사 다음엔 명사보다 동사가 나올 확률이 높음을 알고 판단
# 3. at입력 -> at과 기억된 work(I에 영향을 받은 work)를 연결시켜 tanh... 동일
# 4. google입력 -> google과 기억된 at(I+work에 영향을 받은 at)을 연결시켜 tanh... 동일
# 이처럼 RNN에서는 순환구조를 이용해 기억해 둔 과거의 데이터와 새롭게 입력된 데이터를 연결시켜서
# 순서가 있는 데이터를 해석하는 특징을 가진다.

# 시간 개념을 포함한 RNN 구조
# 순환 구조를 은닉층에서 기억하는 과거의 데이터와 일정시간이 지난 후에 입력되는 데이터에
# 연결시켜 주는 구조로 바꾸어서 생각해볼 수 있음.
# 즉, 문장이나 음성 같은 순서가 있는 데이터라는 것은, 시간의 경과에 따라서 데이터가 순차적으로 들어온다는
# 것과 같은 의미라는 것을 알 수 있음.

# 36강 시작!
# RNN에서의 가중치, 바이어스 동작원리(정량적분석)

# 일반 신경망 아키텍처
# 일반 신경망은 은닉층 A1*W = Z2 -> relu(Z2+b2) = A2
# 출력층 A2*W = Z3 -> relu(Z3+b3) = A3
# 과 같이 바이어스는 각 층(layer)에서 오직 1개의 값으로 정의될 수 있으나, 가중치는
# 각 층(layer)으로 입력되는 데이터의 개수만큼 정의되는 것을 알 수 있음.

# RNN 아키텍처
# 은닉층 바이어스 bh 출력층 바이어스 bo
# 바이어스는 각각의 층(layer)마다 1개씩 있어야 하므로 [은닉층]에서의 바이어스 bh그리고
# 그리고 [출력층] 바이어스 bo 이렇게 바이어스는 총 2개가 있다는것을 알 수 있음
# 은닉층 가중치 Wih, Whh / 출력층 가중치 Who
# [은닉층] 입력되는 데이터, A1에 대해 적용되는 가중치는 Wih이며 은닉층 내부적으로
# 순환구조를 이용하여 기억하고 있는 과거 데이터 H에 적용되는 가중치는 Whh로 정의함
# [출력층] 입력 데이터 A2에 적용되는 가중치는 Who 1개만 있다는 것을 알 수 있음

# RNN 동작원리 - 정략적 분석
# 첫번째 입력데이터 A1에 대한 RNN 동작원리
# 은닉층 A1*Wih = Z2 -> Z2 + Hcur*Whh + bh = R2 (첫 은닉층에선 Hcur=0) 
# -> tanh(R2) = A2 (이때 A2는 다음 입력값에서의 Hcur가 된다.)
# 출력층 A2*Who = Z3 -> softmax(Z3 + b3) = A3
# 두번째 입력데이터 A1에 대한 RNN 동작원리
# 은닉층 A1*Wih = Z2 -> Z2 + Hcur*Whh + bh = R2 
# -> tanh(R2) = A2 (이때 A2는 다음 입력값에서의 Hcur가 된다.)
# 출력층 A2*Who = Z3 -> softmax(Z3 + b3) = A3
# 이때 Z2 + Hcur * Whh + bn = R2 계산을 summation이라고 한다.

# RNN은 순환구조를 가지고있어서 순서가 있는 데이터를 처리할 때 효과적임을 알아보았다.
# 시간 개념을 포함한 현재상태(current state)로 표현해본다면 다음과 같이 나타낼 수 있음
# Ht = A2 = tanh(A1*Wih + H(t-1)*Whh + bh)
# Ht(현재 입력데이터 A1에 대한 state)
# A1(현재 입력데이터 A1)
# Wih(현재 입력데이터 A1에 적용되는 가중치)
# H(t-1)(과거(이전) 입력데이터 A1에 대한 state)
# Whh(과거(이전) state에 적용되는 가중치)
# bh(은닉층 바이어스)

# 37강 시작!
# Tensorflow를 이용한 RNN 예제 구현
# RNN 동작원리
# 은닉층 A1*Wih = Z2 -> Z2+H(t-1)*Whh + bh = R2 -> tanh(R2) = Ht
# tf.contrib.rnn.BasicRNNCell(...) -> Tensorflow 2.0에서는 tf.keras.layers.SimpleRNNCell(...) 대체 예정
# tf.nn.dynamic_rnn(...) 두가지로 구현가능
# 출력층 A2*Whh = Z3 -> softmax(Z3 + bo) = A3 -> cross entropy
# tf.contrib.seq2seq.sequence_loss(...)로 구현가능

# tensorflow API 사용법
# 은닉층
# cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
# [입력파라미터] hidden_size : 내부 순환구조를 가지고 있는 은닉층에서 one-hot으로 표현되는 출력(Ht)
# 크기(size)를 나타내며, one-hot으로 나타내는 정답 크기와 동일함. 예를들어 문자 'A'를 1로, 문자 'B'를 2로 정의한 후에
# 크기가 4인 one-hot방식으로 나타내보면 'A'=[1,0,0,0], 'B'=[0,1,0,0]으로 나타낼 수 있음. 즉, 정답의 크기가
# 4인 one-hot 으로 표현될 수 있기 때문에 hidden_size 또한 4의 값을 가지게 되어 BasicRNNCell(num_units=4) 형식으로 사용됨
# [리턴값] cell: 입력으로 주어진 hidden_size를 가지는 은닉층 객체 cell을 리턴함
# outputs, _states = tf.nn.dynamic_rnn(cell, x_data, initial_state, dtype=tf.float32)
# [입력파라미터] cell : BasicRNNCell(...) 의 리턴값인 은닉층 객체 cell
# [입력파라미터] x_data : 순서를 가지고 있는 입력데이터, 즉 sequence data로서 placeholder 형태로 주어짐
# [입력파라미터] initial_state : 은닉층 객체인 cell 초기상태로서 일반적으로 zero 값으로 초기화 시킴
# [리턴값] outputs, _states: 은닉층 출력 Ht와 상태를 각각 outputs과 _states로 리턴하지만, 실제 outputs 만이 주로 사용됨
# 출력층
# seq_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=label, weights=weights)
# [입력파라미터] outputs : dynamic_rnn(...)의 리턴값인 outputs. 즉, 은닉층 출력 값 Ht를 나타냄
# [입력파라미터] label : 정답데이터를 나타내며 일반적으로 placeholder 형태로 주어짐
# [입력파라미터] weights : 일반적으로 다음과 같이 1로 초기화된 텐서로 주어짐 tf.ones([batch_size, sequence_length]), batch_size는
# 일반적인 batch_size를 말하며, sequence_length는 입력으로 주어지는 문장. 즉, sequence data 길이를 나타냄
# [리턴값] seq_loss : sequence data에 대한 크로스엔트로피 오차(loss)를 리턴함

# 구현할 예제 ('gohome') 이라는 문장
# 'gohom'이라는 순서가 있는 문장을 주면 RNN 아키텍처가 최종적으로 ohome를 예측할 수 있는지 보는 것
# feedforward 과정을 unfold 하면
# g입력(다음글자 o예측) -> o입력(다음글자 h예측) -> ... -> m입력(다음글자 e예측)

# RNN 노드 / 연산 정의 (입력데이터 gohom => 정답데이터 ohome)
# 학습 데이터를 구성하고 있는 unique 문자를 숫자로 나타낸 후, 이러한 숫자를 one-hot 방식으로
# 변환하는 것이 순서가 있는 데이터를 학습하는 RNN 에서 일반적인 방법임.
# 우리 예제의 학습데이터 'gohome'의 unique문자는 5가지 이므로 0부터4까지의 숫자로 문자를 대응시켜
# 입력데이터를 one-hot 방식으로 표현한다.

import tensorflow as tf
import tensorflow_addons as tfa # 이제는 사용하지 않는 contrib.seq2seq를 사용해주기 위한 애드온
tf.compat.v1.disable_eager_execution()
import numpy as np

idx2char = ['g', 'o', 'h', 'm', 'e'] # g=0,o=1,...
x_data = [[0,1,2,1,3]] # gohom
x_one_hot = [[[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]] # gohom
t_data = [[1,2,1,3,4]] # ohome

num_classes = 5 # 정답 크기, 즉 one-hot 으로 나타내는 크기
input_dim = 4 # one-hot size, 즉 입력값은 0부터 3까지 총 4가지임
hidden_size = 5 # output from the RNN, 5 to directly predict one-hot
batch_size = 1 # one sequence (gohom 한개)
sequence_length = 5 # 입력으로 들어가는 문장 길이 gohom = 5
learning_rate = 0.1

X = tf.compat.v1.placeholder(tf.float32, [None,sequence_length, input_dim])
T = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])

# BasicRNNCell()을 이용하여 은닉층 출력 크기가 5인 은닉층 객체 cell을 생성한 후, 
# dynamic_rnn()을 이용하여 cell 출력값 Ht를 계산하여 outputs으로 리턴함
cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) # 이 때 BasicRNNCell은 BasicLSTMCell()혹은 GRUCell() API로 변견 가능하다.
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.compat.v1.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# seq2seq.sequence_loss()를 이용하여 크로스 엔트로피 손실함수를 계산 한 후에, 손실함수 seq_loss가
# 최소가 되도록 AdamOptimizer 를 이용하여 은닉층과 출력층 각각의 가중치와 바이어스 업데이트를 수행함.
weights = tf.ones([batch_size, sequence_length])
seq_loss = tfa.seq2seq.sequence_loss(logits=outputs, targets=T, weights=weights)
loss = tf.reduce_mean(seq_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

y = prediction = tf.argmax(outputs, axis=2)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(2001):
        loss_val, _ = sess.run([loss,train], feed_dict={X: x_one_hot, T: t_data}) # 손실함수 계산 및 가중치/ 바이어스 업데이트
        result = sess.run(y, feed_dict={X: x_one_hot}) # 입력데이터에 대한 예측(prediction) 수행

        if step%400 == 0:
            print("step = ",step, ", loss =", loss_val, ", prediction = ", result, ", target = ", t_data)

            # print char using dic
            result_str = [idx2char[c] for c in np.squeeze(result)] # one-hot 방식으로 계산된 result를 해당 숫자에 대응하는 문자로 변환
            print("\tPrediction = ", ''.join(result_str))
