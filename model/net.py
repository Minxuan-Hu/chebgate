import torch.nn as nn

from .blocks import Block


class ChebResNet(nn.Module):
    def __init__(
        self,
        classes=10,
        K=(3, 5, 5),
        depth=(7, 7, 7),
        widths=(128, 256, 512),
        drop_rate: float = 0.1,
        lap=0.25,
        realization="streamed",
        gate_mode="on",
        stabilize_cheb: bool = False,
    ):
        super().__init__()
        w1, w2, w3 = widths
        total_blocks = sum(depth)
        dp = (
            [0.0] * total_blocks
            if total_blocks <= 1
            else [drop_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        )
        idx = 0

        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, 3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
        )

        self.l1 = nn.Sequential(
            *[
                Block(
                    w1 if i == 0 else w1,
                    w1,
                    K[0],
                    1 if i == 0 else 1,
                    drop_prob=dp[idx + i],
                    lap=lap,
                    realization=realization,
                    gate_mode=gate_mode,
                    stabilize_cheb=stabilize_cheb,
                )
                for i in range(depth[0])
            ]
        )
        idx += depth[0]

        self.l2 = nn.Sequential(
            *[
                Block(
                    w1 if i == 0 else w2,
                    w2,
                    K[1],
                    2 if i == 0 else 1,
                    drop_prob=dp[idx + i],
                    lap=lap,
                    realization=realization,
                    gate_mode=gate_mode,
                    stabilize_cheb=stabilize_cheb,
                )
                for i in range(depth[1])
            ]
        )
        idx += depth[1]

        self.l3 = nn.Sequential(
            *[
                Block(
                    w2 if i == 0 else w3,
                    w3,
                    K[2],
                    2 if i == 0 else 1,
                    drop_prob=dp[idx + i],
                    lap=lap,
                    realization=realization,
                    gate_mode=gate_mode,
                    stabilize_cheb=stabilize_cheb,
                )
                for i in range(depth[2])
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w3, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
