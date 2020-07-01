#pytofch library import
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# scikit-learn library import
from sklearn.datasets import load_digits
from sklearn import datasets, model_selection

#pandas library import
import pandas as pd

#matplotlib library import
from matplotlib import pyplot as plt
from matplotlib import cm
%matplotlib inline # 이미지를 노트북 안에 출력하도록 함

#MNIST 데이터 읽기
mnist = datasets.fetch_mldata('MNIST original')

#설명변수를 정규화하고 변수에 대입 후 화면에 출력
mnist_data = mnist.data / 255

#데이터프레임 객체로 변환하고 화면에 출력
pd.DataFrame(mnist_data)

#1번째 이미지 출력
#plt.imshow(mnist_data[0].reshape(28, 28), cmap=cm.gray_r)
#plt.show()

#목적변수를 변수에 할당하고 데이터를 화면에 출력
mnist_label = mnist.target

#훈련데이터 건수
train_size = 5000
#테스트데이터 건수
test_size = 500

#데이터 집합을 훈련 데이터와 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(mnist_data, mnist_label, train_size = train_size, test_size=test_size)

# 훈련 데이터 텐서 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

# 테스트 데이터 텐서 변환
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

# 변환된 텐서의 데이터 건수 확인
#print(train_X.shape)
#print(train_Y.shape)

# 설명변수와 목적변수 텐서를 합침
train = TensorDataset(train_X, train_Y)

# 텐서의 첫 번째 데이터를 확인
#print(train[0])

# 미니배치 분할
train_loader = DataLoader(train, batch_size=100, shuffle=True)

# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,128)
        self.fc6 = nn.Linear(128,10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.dropout(x, training=self.training)
        x = self.fc6(x)
        return F.log_softmax(x)
    
# 인스턴스 생성
model = Net()

#오차함수 객체
criterion = nn.CrossEntropyLoss()

#최적화를 담당할 객체
optimizer = optim.SGD(model.parameters(), lr=0.01)

#학습시작
for epoch in range(1000):
    total_loss = 0
    
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        
    # 100회 반복마다 누적 오차 출력
    if (epoch+1) % 100 ==0:
        print(epoch+1, total_loss)

#계산 그래프 구성
test_x, test_y = Variable(test_X), Variable(test_Y)
#출력이 0 혹은 1이 되게 함
result = torch.max(model(test_x).data, 1)[1]
# 모형의 정확도 측정
accuracy = sum(test_y.data.numpy() == result.numpy())/len(test_y.data.numpy())

# 모형의 정확도 측정
accuracy