import torch
from torchvision import models
from algorithm.select_alg import selection
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import ConcatDataset
from relabeled_dataset import RelabeledDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_selected_dataloader(train_ds, args, alpha=0.1):
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).to("cuda")
    model.eval()
    origin_train_loader = get_dataloader(train_ds, batch_size=1024)
    with torch.no_grad():
        confidence = torch.tensor([], device="cuda")
        Yhat = torch.tensor([], device="cuda")
        Y = torch.tensor([], device="cuda")
        for data, target in origin_train_loader:
            data = data.to("cuda")
            target = target.to("cuda")
            prob = torch.softmax(model(data), dim=-1)
            pred = torch.argmax(prob, dim=-1)
            msp = prob[torch.arange(data.shape[0]), pred]

            confidence = torch.cat((confidence, msp), dim=0)
            Yhat = torch.cat((Yhat, pred), dim=0)
            Y = torch.cat((Y, target), dim=0)

        Y = Y.detach().cpu().numpy()
        Yhat = Yhat.detach().cpu().numpy()
        confidence = confidence.detach().cpu().numpy()

        n_samples = Y.shape[0]
        n_calib = int(n_samples * 0.1)
        cal_indices = np.random.choice(n_samples, size=n_calib, replace=False)
        cal_dataset = Subset(train_ds, torch.tensor(cal_indices))

        mask = np.ones(n_samples, dtype=bool)
        mask[cal_indices] = False
        test_indices = np.where(mask)[0]
        test_dataset = Subset(train_ds, torch.tensor(test_indices))

        fdp, power, selection_size, selection_indices = selection(Y, Yhat, confidence, cal_indices, alpha,
                                                                  calib_ratio=0.1,
                                                                  random=True, args=args)
        selected_subset = Subset(test_dataset, torch.tensor(selection_indices))
        selected_acc = evaluate(model, selected_subset)
        print(f"Accuracy on the selected subset: {selected_acc}")

        ai_label = torch.tensor([], dtype=torch.int, device="cuda")
        selected_dataloader = get_dataloader(selected_subset, batch_size=1024, shuffle=False)
        for data, target in selected_dataloader:
            data = data.to("cuda")
            prob = torch.softmax(model(data), dim=-1)
            pred = torch.argmax(prob, dim=-1)
            ai_label = torch.cat((ai_label, pred), dim=0)
        relabeled_ds = RelabeledDataset(selected_subset, ai_label)
        merged_dataset = ConcatDataset([cal_dataset, relabeled_ds])
        return get_dataloader(merged_dataset, batch_size=64, shuffle=True)
def get_dataloader(ds, batch_size=64, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=32)


def evaluate(model, val_loader, device="cuda"):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return correct / total


def train(model, train_loader, val_loader, epochs=15, lr=3e-5, weight_decay=0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # -------------------------
        #   Train for one epoch
        # -------------------------
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

        scheduler.step()

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # -------------------------
        #   Evaluate
        # -------------------------
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        print(f"After finetuning:  Val Acc: {val_acc:.4f}")

    return model
