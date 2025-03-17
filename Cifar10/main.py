import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
import  torchvision
from    matplotlib import pyplot as plt
from    matplotlib.pyplot import imshow
from    Lenet import Lenet5
from    Resnet import ResNet18

def main():
    batchsz = 128

    # 数据加载
    cifar_train = datasets.CIFAR10(
        'cifar',  # 数据集的存储路径
        True,  # 是否为训练集，True 表示加载训练集，False 表示加载测试集
        transform=transforms.Compose([
            transforms.Resize((32, 32)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # 每个通道的均值
                std=[0.229, 0.224, 0.225]  # 每个通道的标准差
            )  # 归一化 有助于加快模型的收敛速度
        ]),
        download=True  # 如果数据集不存在，是否下载数据集
    )
    # 加载训练数据，自动将数据分批次
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10(
        'cifar',  # 数据集的存储路径
        False,  # 是否为训练集，True 表示加载训练集，False 表示加载测试集
        transform=transforms.Compose([
            transforms.Resize((32, 32)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # 每个通道的均值
                std=[0.229, 0.224, 0.225]  # 每个通道的标准差
            )  # 归一化
        ]),
        download=True  # 如果数据集不存在，是否下载数据集
    )
    # 加载测试数据，自动将数据分批次
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = next(iter(cifar_train))  # 获取一个批次的数据
    print('x:', x.shape, 'label:', label.shape)  # 打印数据形状

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU，如果可用
    print(f"Using device: {device}")  # 打印设备信息
    # model = Lenet5().to(device)
    model = ResNet18().to(device)  # 初始化ResNet18模型

    criteon = nn.CrossEntropyLoss().to(device)  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 定义优化器
    print(model)  # 打印模型结构

    for epoch in range(5):  # 训练周期
        # 训练循环
        model.train()  # 设置模型为训练模式
        for batchidx, (x, label) in enumerate(cifar_train):  # 分批次遍历数据
            x, label = x.to(device), label.to(device)  # 将数据移动到GPU

            logits = model(x)  # 前向传播
            loss = criteon(logits, label)  # 计算损失

            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        print(f"Epoch {epoch}, loss: {loss.item()}")  # 打印当前损失

        # 测试循环
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)  # 将数据移动到GPU

                logits = model(x)  # 前向传播
                pred = logits.argmax(dim=1)  # 获取预测结果
                correct = torch.eq(pred, label).float().sum().item()  # 计算正确预测的数量
                total_correct += correct
                total_num += x.size(0)

            acc = total_correct / total_num  # 计算准确率
            print(f"Epoch {epoch}, test acc: {acc}")  # 打印测试准确率

    # 可视化展示预测结果
    dataiter = iter(cifar_test)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # 打印图片
    imshow(torchvision.utils.make_grid(images.cpu()))

    # 打印真实标签
    print('GroundTruth: ', ' '.join('%5s' % cifar_train.dataset.classes[labels[j]] for j in range(batchsz)))

    # 预测
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 打印预测标签
    print('Predicted: ', ' '.join('%5s' % cifar_train.dataset.classes[predicted[j]] for j in range(batchsz)))

if __name__ == '__main__':
    main()
