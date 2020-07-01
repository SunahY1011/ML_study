#와인 분류하기 예제

# Pytorch 라이브러리 import
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Scikit-learn library import
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#pandas library import
import pandas as pd

# Setting Learning Data
# wine data loading 
wine = load_wine()

# 데이터 프레임에 담긴 설명변수 출력
#pd.DataFrame(wine.data, columns=wine.feature_names)

# 목적 변수 데이터 출력
#wine.target

# 설명변수와 목적변수를 변수에 대입
wine_data=wine.data[0:130]
wine_target=wine.target[0:130]

# 데이터 집합을 훈련 데이터와 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = train_test_split(wine_data, wine_target, test_size=0.2)

# 데이터 건수 확인용
print(len(train_X))
print(len(test_X))


#####################Linear Model
#Creating Tensor
#훈련 데이터 텐서 변환
train_X=torch.from_numpy(train_X).float()
train_Y=torch.from_numpy(train_Y).long()

#테스트 데이터 텐서 변환
test_X=torch.from_numpy(test_X).float()
test_Y=torch.from_numpy(test_Y).long()

#텐서로 변환한 데이터 건수 확인
#print(train_X.shape)
#print(train_Y.shape)

#설명변수와 목적변수의 텐서를 합침
train = TensorDataset(train_X, train_Y)

#텐서의 첫 번째 데이터 내용 확인
#print(train[0])

#미니베치로 분할
train_loader = DataLoader(train, batch_size=16, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()         # nn.Module class overiding
        self.fc1 = nn.Linear(13,96)
        self.fc2 = nn.Linear(96,2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
    
model = Net()

######################## AutoEncoder
#Creating Tensor
#훈련 데이터 텐서 변환
train_X=torch.from_numpy(train_X).float()
train_Y=torch.from_numpy(train_Y).long()

#테스트 데이터 텐서 변환
test_X=torch.from_numpy(test_X).float()
test_Y=torch.from_numpy(test_Y).long()

#텐서로 변환한 데이터 건수 확인
#print(train_X.shape)
#print(train_Y.shape)

#설명변수와 목적변수의 텐서를 합침
train = TensorDataset(train_X, train_Y)

#텐서의 첫 번째 데이터 내용 확인
#print(train[0])

#미니베치로 분할
train_loader = DataLoader(train, batch_size=16, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()         # nn.Module class overiding
        self.fc1 = nn.Linear(13,96)
        self.fc2 = nn.Linear(96,96)
        self.fc3 = nn.Linear(96,96)
        self.fc4 = nn.Linear(96,96)
        self.fc5 = nn.Linear(96,96)
        self.fc6 = nn.Linear(96,2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x)
    
model = Net()

# 모형 학습
# 오차함수 객체
criterion = nn.CrossEntropyLoss()

# 최적화를 담당할 객체
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 시작
for epoch in range(300):
    total_loss = 0
    
    # 분할해 둔 데이터를 꺼내옴
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)     # 계산 그래프 구성
        optimizer.zero_grad()                                       # 경사초기화
        output = model(train_x)                                     # 순전파 계산
        print(output)
        ssol = criterion(output, train_y)                           # 오차 계산
        ssol.backward()                                             # 역전파 계산
        optimizer.step()                                            # 가중치 업데이트
        total_loss += ssol.data                                     # 누적 오차 계산
        
    if (epoch+1)%50 == 0:
        print (epoch+1, total_loss)

# 계산 그래프 구성
test_x, test_y = Variable(test_X), Variable(test_Y)

# 출력이 0 혹은 1이 되게 함
result = torch.max(model(test_x).data, 1)[1]

# 모형의 정확도 측정
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())

# 모형의 정확도 출력
accuracy