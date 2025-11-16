import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os

def train_uncertainty_predictor(model, mlp, train_loader, device, args):
    """
    训练不确定性预测模型（内部创建特征数据集）

    Args:
        model: 主模型 (ResNet等)
        mlp: 辅助MLP模型
        train_loader: 训练数据加载器
        device: 设备
        args: 训练参数
    """

    # 内部函数：创建特征数据集
    def create_feature_dataset(model, dataloader, device, save_path=None):
        """预计算并创建特征数据集"""
        model.eval()
        
        all_features = []
        all_correctness = []
        
        with torch.no_grad():
            for data, true_target in tqdm(dataloader, desc="Precomputing features"):
                data, true_target = data.to(device), true_target.to(device)
                
                # 获取特征和预测
                features = model.get_feature(data)
                logits = model.feature2logit(features)
                pred_labels = torch.argmax(logits, dim=1)
                
                # 生成正确性标签
                correctness_labels = (pred_labels == true_target).long().unsqueeze(1)
                
                # 收集数据
                all_features.append(features.cpu())
                all_correctness.append(correctness_labels.cpu())
        
        # 合并所有batch
        features_tensor = torch.cat(all_features, dim=0)
        correctness_tensor = torch.cat(all_correctness, dim=0)
        
        # 创建数据集
        class FeatureDataset(torch.utils.data.Dataset):
            def __init__(self, features, correctness):
                self.features = features
                self.correctness = correctness
                
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return self.features[idx], self.correctness[idx]
        
        feature_dataset = FeatureDataset(features_tensor, correctness_tensor)
        
        return feature_dataset

    # 检查是否有缓存的特征数据集
    cache_path = f"feature_cache_{args.dataset}_{hash(str(args))}.pth"

    print("Creating new feature dataset...")
    feature_dataset = create_feature_dataset(
        model, train_loader, device
    )
    
    # 创建特征数据加载器
    feature_loader = torch.utils.data.DataLoader(
        feature_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=64,
        pin_memory=True
    )

    # 定义优化器和损失函数
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    #criterion = nn.MSELoss()  # 使用均方误差损失
    criterion = nn.CrossEntropyLoss()
    # 训练模式
    mlp.train()
    model.eval()  # 主模型保持评估模式

    print(f"Starting training with {len(feature_dataset)} precomputed samples...")
    
    for epoch in tqdm(range(args.epoch)):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # 使用特征数据加载器进行训练
        for features, correctness_labels in feature_loader:
            features = features.to(device)
            correctness_labels = correctness_labels.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # MLP预测正确性分数
            uncertainty_scores = mlp(features)  # [batch_size, 1]

            # 计算损失
            #print(uncertainty_scores.shape, correctness_labels.shape)
            loss = criterion(uncertainty_scores, correctness_labels.view(-1))

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()

            # 计算准确率 (将分数转换为二分类预测)
            predicted_correct = (uncertainty_scores > 0.5).float()
            correct_predictions += (predicted_correct == correctness_labels).sum().item()
            total_samples += features.size(0)

        # 打印训练信息
        accuracy = 100. * correct_predictions / total_samples
        avg_loss = total_loss / len(feature_loader)

        print(f'Epoch [{epoch + 1}/{args.epoch}], '
              f'Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%')

    # 清理缓存（可选）
    if getattr(args, 'cleanup_cache', False) and os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Cleaned up cache file: {cache_path}")

    return mlp

# 为了向后兼容，可以添加一个包装器
def train_uncertainty_predictor_original(model, mlp, train_loader, device, args):
    """
    原始版本的训练函数（不使用特征数据集）
    保持相同的函数签名，内部实现不同
    """
    # 设置参数，不使用缓存
    args.use_cached_features = False
    args.save_features = False
    
    return train_uncertainty_predictor(model, mlp, train_loader, device, args)