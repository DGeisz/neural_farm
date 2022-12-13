from torch import nn
import torchvision
import torch


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()
        print('Tote')

        self.lam = 0.0051

        # So this is where the res net is.  Cool.
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # DANNY - Not sure why we'd have multiple layers for the projector/
        # projector
        sizes = [2048, 8819, 8192, 8192]

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward_reps(self, y1):
        return self.bn(self.projector(self.backbone(y1)))

    def forward(self, y1):
        reps = self.forward_reps(y1)

        # Cool, so I guess we're going to batch norm prior to fucking with this.
        # empirical cross-correlation matrix
        c = reps.T @ reps

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lam * off_diag
        return loss
