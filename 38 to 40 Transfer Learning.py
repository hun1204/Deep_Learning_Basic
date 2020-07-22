# 38강 시작!
# transfer learning의 개념 필요성 fine tuning 에 대해 알아볼 예정
# MNIST를 위한 CNN아키텍처 리뷰
# MNIST(0~9를 나타내는 흑백이미지) 를 높은 정확도로 인식하기 위해서는,
# 최소 3개의 컨볼루션 층과 1개의 완전연결층이 필요하며,
# 이때 전체 학습에 소요되는 시간은 1개의 CPU환경에서 1시간이 소요됨

# CIFAR-10 이라는 고해상도의 칼라 아미지를 CNN으로 구분하기 위해서는
# 최소 5개이상의 컨볼루션층과 2개 이상의 완전연결층을 이용해 입력으로 주어지는
# 복잡한 이미지의 특징(feature)를 추출하는 학습 과정을 거처야 하며,
# 학습에 소요되는 시간은 1개의 CPU 환경이라면 수백~ 수천시간이 소요될 수도 있다.

# 고해상도 칼라 이미지에 잘 훈련된 사전학습(pre-trained)된 CNN 모델이 있다면,
# 이미 학습되어 있는 CNN 모델의 다양한 파라미터 등을 수정(tunning) 해서 사용함으로서,
# 임의의 값으로 초기화된 파라미터를 처음부터 학습시키는 것에 비해 소요시간을 획기적으로 줄일 수
# 있으며 다양한 이미지 데이터를 짧은 시간에 학습 할 수 있는 장점이 있음.

# 실무에서는 Transfer Learning 사용
# 실무에서는 고해상도 칼라 이미지를 학습하는 경우, CNN 아키텍처를 구축하고 임의의 값으로
# 초기화된 파라미터 값(가중치, 바이어스 등) 들을 처음부터 학습시키지 않고,
# 대신 고해상도 칼라 이미지에 이미 학습되어 있는(Pre-Trained) 모델의 가중치와 바이어스를
# 자신의 데이터로 전달(transfer)하여 빠르게 학습하는 방법이 일반적임
# 이처럼 고해상도 칼라 이미지 특성을 파악하는데 있어 최고의 성능을 나타내는 ResNet, GoogleNet을
# 이용하여 우리가 원하는 데이터에 미세조정 즉, Fine Tuning 으로 불리는 작은 변화만을
# 주어 학습시키는 방법을 Transfer Learning (전이학습)이라고 지칭함.

# Pre-Trained Model (Google Inception Model, MS ResNet Model ...)
# 구글의 인셉션 모델은 CNN과 완전연결층으로 이루어진 깊이가 30층 이상으로 이루어져있으며
# 레스넷은 CNN의 깊이가 152층으로 구성되어있고 SKIP CONNECTION 구조를 통해 대폭 성능을 향상 시켰음

# 39강 시작!
# Google Inception-v3(버젼3)를 이용한 실습
# Transfer Learning(전이학습)이란?
# 고해상도 칼라 이미지를 학습하기 위해서 실무에서는 처음부터 CNN 아키텍처를 구축하고
# 가중치나 바이어스 등을 학습시키는 것이 아니라,
# 고해상도 칼라 이미지 특성을 파악하는데 있어서 최고의 성능을 나타내고 소스까지 공개되어 있는
# Google Inception 모델이나 MS ResNet 등을 이용하고 있는데,
# 이처럼 우리가 분석하고자 하는 데이터에 맞도록 미세한 조정, 즉 Fine Tuning 으로 불리는 작은
# 변화만을 주어 학습시키는 방법을 Transfer Learning(전이학습) 이라고 함

# Transfer Learning 실습모델 - Google Inception-v3
# 컨볼루션층과 완전연결층으로 구성된 inception 모델로 불리는 층이 30층 이상이다.
# 고해상도의 칼라이미지를 97% 이상의 정확도로 분석하기 위해 사전학습 되어있다.
# 분석할 데이터는 꽃을 분류하기 위한 daisy, dandelion, rose, sunflower, tulip

# TensorFlow Hub 설치해야하고 
# retrain.py 다운로드
# 고해상도 칼라이미지의 기본특징들이 이미 학습되어 있는 Google Inception-v3 소스 retain.py
# flower_photos.tgz 다운로드 및 압축해제
# Transfer Learning 실습을 위하여 Google에서 기본적으로 제공하는 꽃(flower) 분류 Training Data
# label_image.py 다운로드
# Training data로 학습을 마친 후, 임의의 이미지를 분류하고 정확도를 확인하는 label_image.py

# Transfer Learning 실습 순서
# Training Data (flower_photos) 이용하여 파인튜닝 수행
# 다음과같이 python ./retrain.py --image_dir=./flower_photos 이용하여 Training Data를 재학습 시키면,
# 파인튜닝으로 학습된 가중치와 바이어스등의 학습결과는 /tmp 에 저장됨
# =>
# 학습된 내용을 바탕으로 이미지 분류를 수행해주는 label_image.py를 다음과 같이 실행
# python ./label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt
# --input_layer=Placeholder --output_layer=final_result --image=./sunflower5.jpg
# 마지막 ./sunflower5.jpg만 사용할 예정

# 1. server(back-end)에서 retrain.py 실행하여 /tmp에 학습결과 저장
# 2. Client(front-end)에서 label_image.py 실행
# 3. 이미지 데이터 전달
# 4. 예측 결과 리턴

# 40강 시작!
# 현재 버전이 2.x이므로 39강의 실습과 40강의 나만의 데이터를 이용한 transfer learning은
# 1.x 버전 코드가 많아 실행이 불가하다. 2.x 버전의 소스코드가 필요하다.
