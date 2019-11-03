import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image #PIL = Python Image Library
import pickle

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)
# flatten = True 로 이미지가 1차원 넘파이 배열로 저장됨 
# -> 원래 형상인 28*28 크기의 넘파이 배열로 변형(reshape 메서드 사용) 
# -> 넘파이로 저장된 데이터를 PIL용 데이터 객체로 변환(Image.fromarray 메서드 사용)

img = x_train[0]
label = t_train[0]
print(label) # 5

print(img.shape) # (784,)
img = img.reshape(28,28) 
print(img.shape) # (28,28)

img_show(img)

def softmax(x):
    return x

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f: # rb 읽기위해 2진법 파일을 열고 f로 나타냄
        network = pickle.load(f) # 한 줄씩 파일을 읽어옴

    return network
    # pickle은 프로그램 실행 중 특정 객체를 파일로 저장하는 기능

def predict(network, x): # x는 행렬
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] 

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
  
x, t = get_data()
network = init_network()

accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accurary_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))