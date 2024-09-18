import torch.nn as nn

class HeadBase(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 **kwargs):
        self.num_classes = num_classes
        self.in_c = in_channels
        super().__init__(**kwargs)

        self.fc = nn.Linear(in_channels, num_classes)

        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc(x)
        return x

    def loss(self, output, labels):
        return self.ce(output, labels)
