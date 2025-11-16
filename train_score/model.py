import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn as nn
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
        # 获取原始模型（如果是DataParallel的话）
        if isinstance(self.net, nn.DataParallel):
            original_net = self.net.module
        else:
            original_net = self.net
        
        # Create separate modules
        featurizer = deepcopy(original_net)
        classifier = deepcopy(original_net.fc)

        # Remove final fc from featurizer
        featurizer.fc = nn.Identity()

        # Move to device
        featurizer = featurizer.to("cuda")
        classifier = classifier.to("cuda")

        return nn.DataParallel(featurizer), nn.DataParallel(classifier)

    def get_feature(self, x):
        return self.featurizer(x)
    
    def get_featurizer(self):
        return self.featurizer

    def feature2logit(self, feature):
        return self.classifier(feature)


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(MLPModel, self).__init__()
        hidden_size = input_size // 8
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            #nn.softmax()  # 输出0-1之间的概率
        )

    def forward(self, x):
        return self.network(x)