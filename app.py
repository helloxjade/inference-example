import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# 定义神经网络模型（确保与训练时的结构一致）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 120)  # 确保与训练时的输入维度一致
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 14 * 14)  # 确保与训练时的输出维度一致
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CIFAR-10数据集的类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义数据预处理（确保与训练时一致）
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载CIFAR-10测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# 创建神经网络实例并加载训练好的模型
net = Net()
checkpoint = torch.load('cifar_net.pth')
net.load_state_dict(checkpoint)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 推理函数
def run_inference(testloader, classes):
    net.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        images = images.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        predicted_classes = [classes[predicted[j]] for j in range(len(predicted))]

        return predicted_classes

@app.route('/inference', methods=['GET'])
def inference():
    result = run_inference(testloader, classes)
    return jsonify({'Predicted': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
