# CVAE
CVAE Image generator, implementation relies solely on NumPy
一个非常轻量级的CVAE人脸条件生成模型，训练数据来自Celeba

模型大小只有7.5M左右，前向推理不依赖于Pytorch，使用Numpy实现，且实现方式与原生Pytorch高度相似。

### 效果展示：
![image](https://github.com/hammershock/CVAEGenerator/assets/109429530/429fd044-f71f-4e42-9bfb-7dcd84d49fd6)

```bash
pip install numpy
pip install matplotlib
```
