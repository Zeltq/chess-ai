import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        input_channels=19,
        num_actions=4672,
        board_size=8,
        channels=224,
        num_res_blocks=8,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.board_size = board_size
        self.channels = channels
        self.num_res_blocks = num_res_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        policy_conv_channels = 2
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_conv_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_conv_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(policy_conv_channels * board_size * board_size, num_actions),
        )

        value_conv_channels = 1
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_conv_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_conv_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_conv_channels * board_size * board_size, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_config(self):
        return {
            "input_channels": self.input_channels,
            "num_actions": self.num_actions,
            "board_size": self.board_size,
            "channels": self.channels,
            "num_res_blocks": self.num_res_blocks,
        }

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        return self.policy_head(x), self.value_head(x)
