# 41강 시작!
# 코랩으로 불리는 Google Colaboratory의 개념, 장점, 사용법

# 딥러닝 개발환경 구축(standalone)
# 보통 TensorFlow, Keras, PyTorch 등 딥러닝 나이브러리를 개인 PC에 설치하고 개발과 테스트를 수행한다.
# 병렬처리가 가능한 GPU를 사용하면 개발과 테스트 수행 시 성능을 높일 수 있다.

# standalone 방식의 문제점
# 호환성 문제로 인해서 라이브러리가 설치되지 않을 수 있음
# -파이썬 3.7버전 이상에서 TensorFlow가 설치되지 않는 경우,2019 1월 기준
# 딥러닝 라이브러리 간의 종속적인 관계(dependency)를 파악해야 함
# -Transfer Learning에 필요한 TensorFlow-Hub 라이브러리는 TensorFlow 버전 1.7 이상에서만 설치됨
# 개발과 테스트 성능을 높이기 위해서 GPU 같은 고가의 하드웨어를 별도 구매해야 함.

# 딥러닝 개발을 쉽게 하는 법 - Google Colaboratory
# 1. 딥러닝 개발을 위한 라이브러리가 이미 설치되어 있고,
# 2. 고가인 GPU도 저렴하게 또는 무료로 이용할 수 있으며,
# 3. GitHub 등과의 연동을 통해서 자유롭게 소스를 올리고 가져올 수 있는 개발 환경
# -> standalone방식 문제점 해결(Google Colaboratory - Colab)

# Google Colaboratory(Colab)은
# gmail 계정이 있는 개발자라면 누구나 무료로 사용할 수 있는 클라우드 서비스로서,
# 파이썬과 TensorFlow, Keras 등의 딥러닝 라이브러리 등이 미리 설치되어 있기 때문에,
# 웹 브러우저만으로 주피터(Jupyter Notebook) 작업을 할 수 있으며, 무엇보다도 가장
# 매력적인 것은 GPU를 무료로 사용할 수 있다는 점이다.
# 그리고 Google Colab을 통해서 Google Drive나 GitHub 연동을 통해서 자료 공유와 협업을
# 쉽게 할수 있다는 장점이 있다.

# https://colab.research.google.com 에 접속후 gmail계정으로 login함
# 코드, 텍스트 추가 가능하며 한 라인마다 왼쪽 실행버튼 혹은 shift+enter를 통해 사용

# 시스템사양 (System Spec.)
# Jupyter Notebook 에서 cat, ls, head, wget 등의 리눅스 쉘 명령어를 사용하기 위해서는
# 명령문 앞에 느낌표(!) 를 붙이면 Jupyter Notebook 에서 바로 실행 할 수 있음

# 나머지는 Colab에서 직접 실습!
