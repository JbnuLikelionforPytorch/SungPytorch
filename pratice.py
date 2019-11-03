import numpy as np

def relu(x):
    return np.maximum(0,x) # ReLU 함수

def sigmoid(x):
    return 1/ (1+ np.exp(-x)) # sigmoid 함수
# -----------------------------------------------------------------------------------------------------------------------------
x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])

print(x.shape)
print(w1.shape)
print(b1.shape)

a1 = np.dot(x, w1) + b1
w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])

z1 = a1

a2 = np.dot(z1, w2) + b2
z2 = sigmoid(a2)

def idenity_function(x): # 항등함수
    return x 

w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

a3 = np.dot(w3, z2) + b3
print(a3)

# ------------------------------------------------------------------------------------------------------------------------------
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]]) # 2*3행렬
    network['b1'] = np.array([0.1, 0.2, 0.3]) #bias 3개
    
    network['W2'] = np.array([[0.1, 0.4],[0.2, 0.5], [0.3, 0.6]]) # 3*2행렬
    network['b2'] = np.array([0.1, 0.2]) #bias 2개
    
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2*2행렬
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = idenity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])  #초기 입력값
y = forward(network, x)
print(y)