# 使用官方 PyTorch 基础镜像
FROM registry.xxxx.com:5000/test-xy/pytorch:latest

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到容器中的工作目录
COPY . /app

COPY ./data /app/data
# 安装 Flask, torch 和 torchvision
RUN pip install Flask torch torchvision

# 下载数据集到镜像中

# 提前复制模型参数文件到镜像中
EXPOSE 5000


# 运行 Flask 应用
CMD ["python", "app.py"]

