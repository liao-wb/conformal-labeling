import torch.nn as nn
import torch.optim as optim
import torch

def train_uncertainty_predictor(model, mlp, train_loader, device, args):
    """
    训练不确定性预测模型

    Args:
        model: 主模型 (ResNet等)
        mlp: 辅助MLP模型
        train_loader: 训练数据加载器
        device: 设备
        args: 训练参数
    """

    # 定义优化器和损失函数
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()  # 使用均方误差损失

    # 训练模式
    mlp.train()
    model.eval()  # 主模型保持评估模式

    for epoch in range(args.epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for data, true_target in train_loader:
            data, true_target = data.to(device), true_target.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播 - 获取主模型的特征和预测
            with torch.no_grad():
                features = model.get_feature(data)  # 获取特征 [batch_size, 512]
                logits = model.feature2logit(features)
                pred_labels = torch.argmax(logits, dim=1)

                # 生成正确性标签: 1=正确, 0=错误
                correctness_labels = (pred_labels == true_target).float().unsqueeze(1)  # [batch_size, 1]

            # MLP预测正确性分数
            uncertainty_scores = mlp(features)  # [batch_size, 1]

            # 计算损失
            loss = criterion(uncertainty_scores, correctness_labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()

            # 计算准确率 (将分数转换为二分类预测)
            predicted_correct = (uncertainty_scores > 0.5).float()
            correct_predictions += (predicted_correct == correctness_labels).sum().item()
            total_samples += data.size(0)

        # 打印训练信息
        accuracy = 100. * correct_predictions / total_samples
        avg_loss = total_loss / len(train_loader)

        print(f'Epoch [{epoch + 1}/{args.epochs}], '
              f'Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%')

    return mlp