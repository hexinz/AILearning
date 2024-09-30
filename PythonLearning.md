# PythonLearning
## 环境配置
下载Anaconda
```
conda create -n openai-demo python=3.8
conda activate openai-demo
```
```
pip install openai -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install --trusted-host mirrors.tools.huawei.com torch torchvision torchaudio -i https://mirrors.tools.huawei.com/pypi/simple
```
## Pytorch
### Einsum
https://blog.csdn.net/zhaohongfei_358/article/details/125273126
```
A = torch.Tensor(range(2,3,4)).view(2,3,4)
C = torch.einsum(“ijk->jk”, A)
```
