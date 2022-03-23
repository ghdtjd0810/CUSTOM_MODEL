!pip install einops
import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda
from torch import Tensor
!pip install timm

from torchsummary import summary

###
# 현재 이미지 64에 58% 이미지 32에 59% 128이면?
# (인코더층은 상관없음)

# 현재 이미지 64에 층 12에 패치사이즈 8

# 인코더 층 건드리기. patch 사이즈 수정하기

# 현재, 데이터 어그멘테이션 64사이즈에 패치 8이고 아이덴티티 행렬, 정방행렬 더해줄 필요도 있음. 
###


#currently img, 32, batch 30

patch_size = 8
img_size = 64
dim_size = 512
num_channel = 3

import gc
gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualAdd(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn#.to(device)

  def forward(self, x, **kwargs):
    res = x
    x = self.fn(x, **kwargs)
    x += res
    return x

class FFNN(nn.Sequential):
  def __init__(self, dim_size: int , expansion: int = 4, dropout: float = 0.):
    super().__init__(
        nn.Linear(dim_size,expansion * dim_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expansion* dim_size, dim_size)
    )
    #self.dim_size = dim_size
    #self.h_dim_size = h_dim_size

  #def forward(self, x):
   # return self.ffnn(x)


class Cus_Embedding(nn.Module):
  def __init__(self, patch_size : int = 8, img_size: int =128, dim_size: int = 512, num_channel: int = 3):
    super().__init__()

    self.patch_size = patch_size
    self.img_size = img_size
    self.dim_size = dim_size
    self.num_channel = num_channel

    self.num_patches = (img_size * img_size)// patch_size**2 # 256
    self.embedding_size = patch_size * patch_size * num_channel # 768

    self.cls_token = nn.Parameter(torch.randn(50,1,dim_size))#.to(device) #(1,1,512)
    self.positional_embedding = nn.Parameter(torch.randn(1,self.num_patches +1, dim_size))#.to(device) #( 1, 257, 512)
    self.sample_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    self.linear = nn.Linear(self.embedding_size, self.dim_size)#.to(device)
  def forward(self,x: Tensor) -> Tensor:

    x = self.sample_embedding(x)
    b,n,_ = x.shape
    #linear = nn.Linear(self.embedding_size, self.dim_size).to(device)
    x = self.linear(x)
    x = torch.cat((self.cls_token, x), dim = 1)
    x = x + self.positional_embedding[:,:(n+1)]

    return x


class Cus_MHA(nn.Module):
  def __init__(self, d_model: int = 512, dim_heads: int =64, num_heads: int = 8):
    super().__init__()
    self.d_model = d_model
    self.dim_heads = dim_heads
    self.num_heads = num_heads
    #여기 고쳤음 밑에 #치고
    self.qkv = nn.Linear(self.d_model, self.d_model*3)#.to(device)

    self.linear = nn.Sequential(
        nn.Linear(self.d_model, self.d_model)#.to(device)
    )

  def forward(self, x):
    #qkv = nn.Linear(self.d_model, self.d_model*3).to(device)
    #linear = nn.Linear(self.d_model, self.d_model).to(device)

    qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d', qkv = 3, h = self.num_heads)
    query, key, value = qkv[0], qkv[1], qkv[2]
    dot_matmul = torch.matmul(query, key.transpose(-1,-2))
    softmax = nn.Softmax(dim=-1)
    scaling = self.dim_heads ** 0.5
    dot_matmul_sof = softmax(dot_matmul)/ scaling
    dot_matmul_v = torch.matmul(dot_matmul_sof, value)
    final_MHA = rearrange(dot_matmul_v, 'b h e d -> b e (h d)')
    out = self.linear(final_MHA)
    return out


class Transformerblock(nn.Sequential):
  def __init__(self, d_model: int = 512,foward_expansion : int= 4, forward_drop_p: float= 0.,drop_p: float =0. ,**kwargs ):
    super().__init__(
        ResidualAdd(nn.Sequential(
            nn.LayerNorm(d_model),#.to(device),
            Cus_MHA(d_model, **kwargs),
            nn.Dropout(drop_p)
        )),
        ResidualAdd(nn.Sequential(
           nn.LayerNorm(d_model),#.to(device)
           FFNN(dim_size, expansion = foward_expansion, dropout = drop_p),
           nn.Dropout(drop_p)
        ))
    )

class TransformerEncoder(nn.Sequential):
  def __init__(self, depth : int = 12, **kwargs):
    super().__init__(*[Transformerblock(**kwargs) for _ in range(depth)])

class classfyhead(nn.Sequential):
  def __init__(self, d_model = 512, n_classes = 10):
    super().__init__(
        Reduce('b n e -> b e', reduction = 'mean'),
        nn.LayerNorm(d_model),#.to(device),
        nn.Linear(d_model, n_classes)#.to(device)
    )

class ViT(nn.Sequential):
  def __init__(self,
               num_channel: int = 3,
               patch_size: int = 8,
               img_size: int = 128,
               dim_size: int = 512,
               d_model: int = 512,
               depth: int = 12,
               n_classes: int = 10,
               **kwargs):
    super().__init__(
        Cus_Embedding(patch_size , img_size , dim_size , num_channel),
        TransformerEncoder(depth, d_model = d_model, **kwargs),
        classfyhead(d_model, n_classes)
    )

model = ViT()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-5)

model.to(device)



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  
epochs = 1

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, model, criterion, optimizer)
    test(testloader, model,criterion)
    torch.save(model.state_dict(), 'model_cus64.pth')
