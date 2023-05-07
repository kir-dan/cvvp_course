import torch


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x):
        return x


class Classifier:
    def __init__(self, model_path):
        self.model = torch.load(model_path).cuda()
        self.model.eval()

    def calculate_descriptors(self, imgs):
        return self.model(imgs)
