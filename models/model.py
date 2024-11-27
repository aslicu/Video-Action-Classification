import torchvision
import torch.nn as nn
import torch.nn.functional as F

class VideoR3D18(nn.Module):
    def __init__(self, num_classes, embed_dim=400, proj_dim=128, proj_dim2=400):
        super(VideoR3D18, self).__init__()
        self.base_model = torchvision.models.video.r3d_18(pretrained=False)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, embed_dim)

        # Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.classifier = self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        embed = self.base_model(x)
        proj_embed = self.projection_head(embed)
        proj_embed = F.normalize(proj_embed, p=2, dim=1)
        output = self.classifier(embed)

        return output, proj_embed



