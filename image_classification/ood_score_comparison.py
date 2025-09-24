import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import argparse
import pandas as pd
from torchvision.models import ResNet34_Weights, DenseNet161_Weights, ResNeXt50_32X4D_Weights
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--model", type=str, default="resnet34")
parser.add_argument("--temperature", type=float, default=1000.0, help="temperature scaling")
parser.add_argument("--epsilon", type=float, default=0.0014, help="perturbation magnitude")
parser.add_argument("--react_threshold", type=float, default=1.0, help="ReAct threshold for feature clipping")
args = parser.parse_args()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = args.model

# 定义特征提取的hook
features = {}


def get_features(name):
    def hook(model, input, output):
        features[name] = output

    return hook


if model_name == "resnet34":
    model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device).eval()
    # 注册hook到倒数第二层（全连接层之前）
    model.fc.register_forward_hook(get_features('penultimate'))
elif model_name == "densenet161":
    model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1).to(device).eval()
    # DenseNet的倒数第二层是classifier之前
    model.classifier.register_forward_hook(get_features('penultimate'))
elif model_name == "resnext50":
    model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1).to(device).eval()
    model.fc.register_forward_hook(get_features('penultimate'))
else:
    raise NotImplementedError(f"Model {model_name} not supported")

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if args.dataset == "imagenet":
    test_dataset = torchvision.datasets.ImageFolder(
        root="/mnt/sharedata/ssd_small/common/datasets/imagenet/images/val",
        transform=val_transform
    )
elif args.dataset == "imagenetv2":
    test_dataset = torchvision.datasets.ImageFolder(
        root="/mnt/sharedata/ssd_small/common/datasets/imagenetv2/imagenetv2-matched-frequency-format-val",
        transform=val_transform
    )
else:
    raise NotImplementedError(f"Dataset {args.dataset} not supported")

# For ImageNetV2, remap labels to match ImageNet class indices (0-999)
label_remap = None
if args.dataset == "imagenetv2":
    class_names = test_dataset.classes
    label_remap = {sorted_idx: int(class_name) for sorted_idx, class_name in enumerate(class_names)}
    print(f"Applied label remapping for ImageNetV2. Mapping size: {len(label_remap)}")

dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

# Initialize lists to store results
all_msp_confidences = []
all_odin_confidences = []
all_energy_confidences = []
all_react_confidences = []  # 新增ReAct scores
all_max_logits = []  # 新增Max Logit scores
all_y_hat_odin = []
all_y_hat = []
all_y_true = []

temperature = args.temperature
epsilon = args.epsilon
react_threshold = args.react_threshold
criterion = torch.nn.CrossEntropyLoss()


def compute_react_score(model, features, threshold=1.0):
    """
    计算ReAct score
    features: 倒数第二层的特征
    threshold: 裁剪阈值
    """
    # 裁剪特征（ReAct的核心操作）
    clipped_features = torch.clamp(features, max=threshold)

    # 对于不同的模型结构，需要不同的处理方式
    if model_name == "resnet34" or model_name == "resnext50":
        # ResNet类模型：将裁剪后的特征传入最后的全连接层
        react_logits = model.fc(clipped_features)
    elif model_name == "densenet161":
        # DenseNet：需要全局平均池化 + 全连接层
        react_logits = model.classifier(clipped_features)
    else:
        raise NotImplementedError(f"ReAct not implemented for {model_name}")

    # 使用MSP作为score
    react_probs = torch.softmax(react_logits, dim=-1)
    react_scores = react_probs.max(dim=-1)[0]
    return react_scores


for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    data.requires_grad = True

    # Forward pass with temperature scaling
    logits = model(data)

    # 获取特征（通过hook）
    penultimate_features = features['penultimate']

    # 计算各种scores
    prob = torch.softmax(logits, dim=-1)
    y_hat_msp = torch.argmax(prob, dim=-1)
    msp_conf = prob[torch.arange(prob.size(0)), y_hat_msp]

    # Energy score
    energy_conf = torch.logsumexp(logits, dim=-1)

    # Max Logit score
    max_logit = torch.max(logits, dim=-1)[0]

    # ReAct score
    react_conf = compute_react_score(model, penultimate_features, react_threshold)

    # 存储基础scores
    all_msp_confidences.extend(msp_conf.detach().cpu().numpy())
    all_energy_confidences.extend(energy_conf.detach().cpu().numpy())
    all_max_logits.extend(max_logit.detach().cpu().numpy())
    all_react_confidences.extend(react_conf.detach().cpu().numpy())
    all_y_hat.extend(y_hat_msp.detach().cpu().numpy())

    # ODIN计算（保持不变）
    logits_temp = logits / temperature
    pred = torch.argmax(logits_temp, dim=1)
    msp_temp = torch.softmax(logits_temp, dim=-1)[torch.arange(logits_temp.size(0)), pred]

    loss = torch.sum(torch.log(msp_temp))
    model.zero_grad()
    loss.backward()

    # Perturbation
    gradient = data.grad.data
    perturbation = epsilon * torch.sign(-gradient)
    data_perturbed = data - perturbation
    data_perturbed = torch.clamp(data_perturbed, 0, 1)

    # Forward again with perturbed input
    logits_perturbed = model(data_perturbed) / temperature
    prob_perturbed = torch.softmax(logits_perturbed, dim=-1)
    y_hat_odin = torch.argmax(prob_perturbed, dim=-1)
    conf_odin = prob_perturbed[torch.arange(prob_perturbed.size(0)), y_hat_odin]

    # Store ODIN results
    all_odin_confidences.extend(conf_odin.detach().cpu().numpy())
    all_y_hat_odin.extend(y_hat_odin.cpu().numpy())

    # Remap targets if needed
    batch_targets = target.cpu().numpy()
    if label_remap is not None:
        batch_targets = np.array([label_remap[t.item()] for t in target])
    all_y_true.extend(batch_targets)

# Create DataFrame with all scores
df = pd.DataFrame({
    'Y': all_y_true,
    "Yhat": all_y_hat,
    'Yhat_odin': all_y_hat_odin,
    'odin_confidence': all_odin_confidences,
    "msp_confidence": all_msp_confidences,
    "energy_confidence": all_energy_confidences,
    "react_confidence": all_react_confidences,  # ReAct scores
    "max_logit": all_max_logits  # Max Logit scores
})

# Save to CSV
output_file = f'{args.model}_{args.dataset}_oodscore.csv'
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
print(f"ReAct threshold used: {react_threshold}")