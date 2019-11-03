import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False) #normalize 입력 이미지의 픽셀값을 0.0 ~ 1.0 사이의 값으로 정규화할지 정함
                                              #flattern 입력이미지를 1차원 배열로 만들지를 정함
# 5번째 줄 어떤 구조인지 모르겠음 - (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형식 반환

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)