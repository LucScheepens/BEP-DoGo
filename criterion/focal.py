import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import positive_mask


class BalancedFocalLoss(nn.Module):
    def __init__(self, args):
        super(BalancedFocalLoss, self).__init__()
        self.batch_size = args.train.batchsize
        self.temperature = args.train.temperature
        self.gamma = args.train.loss.gamma
        self.alpha = args.train.loss.alpha
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.N = 2 * self.batch_size
        self.mask = positive_mask(args.train.batchsize)


    def forward(self, zx, zy):
        positive_samples, negative_samples = self.sample_no_dict(zx, zy)
        labels = torch.zeros(self.N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        ce_loss = self.criterion(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        balanced_focal_loss = self.alpha * focal_loss

        return balanced_focal_loss.mean()
    
    def sample_no_dict(self, zx, zy):
        """
        Positive and Negative sampling without dictionary
        """
        z = torch.cat((zx, zy), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # Since projections are already normalized using F.normalize,
        # below function can be used instead of CosineSimilarity
        # sim = torch.div(torch.matmul(z, z.T), self.temperature)

        # Extract positive samples
        sim_xy = torch.diag(sim, self.batch_size)
        sim_yx = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_xy, sim_yx), dim=0).reshape(self.N, 1)

        # Extract negative samples
        negative_samples = sim[self.mask].reshape(self.N, -1)
        return positive_samples, negative_samples
