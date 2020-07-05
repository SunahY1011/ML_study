# OS Library Import
import os
# 정규표현식 library import
import re

# scikit-learn library import
from sklearn import datasets, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 형태소 분석기 라이브러리
from konlpy.tag import Hannanum
from konlpy.tag import Kkma

# pandas library import 
import pandas as pd

# numpy library import
import numpy as np

#-*-coding utf-8-*-
# 데이터 정리
dir_prefix = './data/'
target_dir = 'HKIB-20000'
cat_dirs = ['health', 'economy', 'science', 'education', 'culture', 'society', 'industry', 'leisure', 'politics']
cat_prefixes = ['건강', '경제', '과학', '교육', '문화', '사회', '산업', '여가', '정치']

files = os.listdir(dir_prefix+target_dir)

# 5분할된 텍스트 파일을 각각 처리
for file in files:
    # 데이터가 담긴 파일만 처리
    if not file.endswith('.txt'):
        continue
    
    # 각 텍스트 파일을 처리
    with open(dir_prefix + target_dir + '/' + file,"r", encoding='utf-8') as currfile:
        doc_cnt = 0
        docs = []
        curr_doc = None
        
        # 기사 단위로 분할해서 리스트를 생성
        for curr_line in currfile:
            if curr_line.startswith('@DOCUMENT'):
                if curr_doc is not None:
                    docs.append(curr_doc)
                curr_doc = curr_line
                doc_cnt = doc_cnt + 1
                continue
            curr_doc = curr_doc + curr_line
            
        # 각 기사를 대주제별로 분류해서 기사별 파일로 정리
        for doc in docs:
            doc_lines = doc.split('\n')
            doc_no = doc_lines[1][9:]
            
            # 주제 추출
            doc_cat03 = ''
            for line in doc_lines[:10]:
                if line.startswith("#CAT'03:"):
                    doc_cat03 = line[10:]
                    break
                    
            # 추출한 주제별로 디렉터리 정리
            for cat_prefix in cat_prefixes:
                if doc_cat03.startswith(cat_prefix):
                    dir_index = cat_prefixes.index(cat_prefix)
                    break
                    
            # 문서 정보를 제거하고 기사 본문만 남기기
            filtered_lines = []
            for line in doc_lines:
                if not (line.startswith('#') or line.startswith('@')):
                    filtered_lines.append(line)
            
            # 주제별 디렉터리에 기사를 파일로 쓰기
            filename = 'hkib-' + doc_no + '.txt'
            filepath = dir_prefix + target_dir + '/' + cat_dirs[dir_index]
            
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            f = open(filepath + '/' + filename, "w", encoding="utf-8")
            f.write("\n".join(filtered_lines))
            f.close()
# 대상이 되는 주제 폴더 선택 (경제, 사회)
dirs = ['education', 'health']

# 각 폴더의 파일을 하나씩 읽어 들임
for i, d in enumerate(dirs):
    # 파일 목록 읽어오기
    files = os.listdir(dir_prefix + target_dir +'/' + d)
    
    for file in files:
        # 각 파일을 읽어 들이기
        f = open(dir_prefix + target_dir +'/' + d + '/' + file, "r", encoding = "utf-8")
        raw = f.read()
    
        # 정규표현식을 사용해 불필요한 문자열을 제거한 다음 파일 내용을 출력
        reg_raw = re.sub(r'[0-9a-zA-Z]', '', raw)
        reg_raw = re.sub(r'[-\'@#:/◆<>!-"*\(\)]', '', raw)
        reg_raw = re.sub(r'[ ]+', '', reg_raw)
        reg_raw = reg_raw.replace('\n', ' ')
    
        # 파일 닫기
        f.close()

# 기사에 출현하는 단어와 레이블을 저장할 리스트를 생성
# 설명변수
x_ls = []
# 목적변수
y_ls = []
tmp1 = []
tmp2 = ''

# 형태소 분석기 객체 생성
# tokenizer_han = Hannanum()
tokenizer = Kkma()

# 각 폴더의 파일을 하나씩 읽어 들이며, 전처리 후 리스트에 저장
for i, d in enumerate(dirs):
    # 파일 목록 읽어오기
    files = os.listdir(dir_prefix + target_dir + '/' + d)
    
    for file in files:
        # 각 파일을 읽어 들이기
        f = open(dir_prefix + target_dir + '/' + d + '/' + file, 'r', encoding='utf-8')
        raw = f.read()
        
        # 정규표현식을 사용해 불필요한 문자열을 제거한 다음 파일 내용을 출력
        reg_raw = re.sub(r'[-\'@#:/◆<>!-"*\(\)]', '', raw)
        reg_raw = re.sub(r'[ ]+', '', reg_raw)
        reg_raw = reg_raw.replace('\n', ' ')
        
        # 형태소 분석을 거쳐 명사만 추출한 리스트를 생성
        tokens = tokenizer.nouns(reg_raw)                     #동사 제거
        
        for token in tokens :
            tmp1.append(token)
            
        tmp2 = ' '.join(tmp1)
        x_ls.append(tmp2)
        tmp1 = []
        
        # 기사 주제 레이블을 리스트에 저장
        y_ls.append(i)
        
        # 파일 닫기
        f.close()

# 데이터프레임으로 변환해서 설명변수를 화면에 출력
#pd.DataFrame(x_ls)

# 첫 번째 기사로부터 추출한 단어를 출력
print(x_ls[0])

# 목적변수를 화면에 출력
print(y_ls)

# 설명변수와 목적변수를 Numpy 배열로 변환
x_array = np.array(x_ls)
y_array = np.array(y_ls)

# 단어 출현 횟수를 계수
cntvec = CountVectorizer()
x_cntvecs = cntvec.fit_transform(x_array)
x_cntarray = x_cntvecs.toarray()

#pd.DataFrame(x_cntarray)

# 단어와 단어의 인덱스 표시
for k, v in sorted(cntvec.vocabulary_.items(), key=lambda x:x[1]):
    print(k, v)

# 단어의 TF-IDF 계산
tfidf_vec = TfidfVectorizer(use_idf=True)            # TF-IDF 계산해주는 객체 생성
x_tfidf_vecs = tfidf_vec.fit_transform(x_array)       # 설명변수에 대한 TF-IDF 계산
x_tfidf_array = x_tfidf_vecs.toarray()                # 벡터로 되어있는 거 array로 바꿔줌

# 데이터프레임으로 변환해서 단어의 출현횟수를 출력
pd.DataFrame(x_tfidf_array)

# 데이터를 훈련 데이터와 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(x_tfidf_array, y_array, test_size=0.2)

# 데이터 건수 확인
print(len(train_X))
print(len(test_X))

# Pytorch Library Import
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 훈련 데이터 텐서 생성
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

# 테스트 데이터 텐서 생성
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

# 텐서로 변환한 데이터 건수 확인
print(train_X.shape)
print(train_X.shape)

# 설명변수와 목적변수의 텐서를 합친다
train = TensorDataset(train_X, train_Y)

# 첫 번째 텐서 내용 확인
print(train[0])

# 미니배치 분할
train_loader = DataLoader(train, batch_size=100, shuffle=True)

# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(48706,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,128)
        self.fc6 = nn.Linear(128,2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc5(x)
        return F.log_softmax(x)
    
# 인스턴스 생성
model = Net()
#model.cuda()

#오차함수 객체
criterion = nn.CrossEntropyLoss()

#최적화를 담당할 객체
optimizer = optim.Adam(model.parameters(), lr=0.005)

#학습시작
for epoch in range(20):
    total_loss = 0
    
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        
    # 각 epoch 반복마다 누적 오차 출력
    print(epoch+1, total_loss)

#계산 그래프 구성
test_x, test_y = Variable(test_X), Variable(test_Y)
#출력이 0 혹은 1이 되게 함
result = torch.max(model(test_x).data, 1)[1]
# 모형의 정확도 측정
accuracy = sum(test_y.cpu().data.numpy() == result.cpu().numpy())/len(test_y.cpu().data.numpy())

# 모형의 정확도 측정
accuracy