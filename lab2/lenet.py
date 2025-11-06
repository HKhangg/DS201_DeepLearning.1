import torch
from torch import nn

class LeNet(nn.Module):
    def __init__ (self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=(5,5),
                stride=1,
                padding=(2,2)
            ),

            nn.Sigmoid(),

            nn.AvgPool2d(
                stride=(2,2),
                kernel_size=(2,2),
                padding=(0,0)
            ),

            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=(5,5),
                padding=0,
            ),

            nn.Sigmoid(),

            nn.AvgPool2d(
                stride=(2,2),
                kernel_size=(2,2),
                padding=(0,0),
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=120,
                kernel_size=(5,5),
                padding=0,
            ),

            nn.Sigmoid(),

            nn.Flatten(
                start_dim=1,
                end_dim=-1,
            ),

            nn.Linear(
                in_features=120,
                out_features=84,
            ),

            nn.Sigmoid(),

            nn.Linear(
                in_features=84,
                out_features=10,
            ),
        )

    def forward(self, images: torch.Tensor) -> float:
        return self.model(images)
    
