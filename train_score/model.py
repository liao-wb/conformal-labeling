import torch.nn as nn
import torch
from copy import deepcopy

class NaiveModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

        self.featurizer, self.classifier = self.get_featurizer_and_classifier()
    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def forward(self, x):
        return self.net(x)

    def tune(self, tune_loader):
        raise NotImplementedError

    def get_featurizer_and_classifier(self):
        # Create separate modules
        featurizer = deepcopy(self.net)
        classifier = deepcopy(self.net.fc)

        # Remove final fc from featurizer
        featurizer.fc = nn.Identity()

        # Move to device
        featurizer = featurizer.to("cuda")
        classifier = classifier.to("cuda")

        return featurizer, classifier

    def get_feature(self, x):
        return self.featurizer(x)
    def get_featurizer(self):
        return self.featurizer

    def feature2logit(self, feature):
        return self.classifier(feature)


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()  # 输出0-1之间的概率
        )

    def forward(self, x):
        return self.network(x)