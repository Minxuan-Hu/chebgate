import torch.nn as nn

from chebgate.core.droppath import DropPath
from .chebconv import ChebConv2d


class Block(nn.Module):
    def __init__(
        self,
        Cin,
        Cout,
        K,
        stride,
        drop_prob: float = 0.0,
        lap=0.25,
        realization="streamed",
        gate_mode="on",
        stabilize_cheb: bool = True,
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(stride) if stride != 1 else nn.Identity()
        self.down = (
            nn.Identity()
            if (stride == 1 and Cin == Cout)
            else nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(Cin, Cout, 1, bias=False),
                nn.BatchNorm2d(Cout),
            )
        )

        self.c1 = ChebConv2d(
            Cin,
            Cout,
            K,
            lap=lap,
            realization=realization,
            gate_mode=gate_mode,
            stabilize_cheb=stabilize_cheb,
        )
        self.b1 = nn.BatchNorm2d(Cout)

        self.c2 = ChebConv2d(
            Cout,
            Cout,
            K,
            lap=lap,
            realization=realization,
            gate_mode=gate_mode,
            stabilize_cheb=stabilize_cheb,
        )
        self.b2 = nn.BatchNorm2d(Cout)

        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        identity = self.down(x)
        x = self.pool(x)
        out = self.act(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        out = self.drop_path(out)
        return self.act(out + identity)
