import torch
import torch.nn as nn
import torch.nn.functional as F # noqa
from torchvision import models
from torchvision.models.resnet import Bottleneck, ResNet


class LeftRightAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape
        assert W % 2 == 0, "Width must be even to split into left/right halves"

        # Split into left and right halves
        left = x[:, :, :, :W // 2]  # (N, C, H, W/2)
        right = x[:, :, :, W // 2:]  # (N, C, H, W/2)

        # Take mean across width dimension for each half
        left_mean = left.mean(dim=(2, 3), keepdim=True)  # (N, C, H, 1)
        right_mean = right.mean(dim=(2, 3), keepdim=True)  # (N, C, H, 1)

        # Concatenate along width dimension
        out = torch.cat([left_mean, right_mean], dim=3)  # (N, C, H, 2)
        return out

class LeftRightRowAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape
        assert W % 2 == 0, "Width must be even to split into left/right halves"

        # Split into left and right halves
        left = x[:, :, :, :W // 2]  # (N, C, H, W/2)
        right = x[:, :, :, W // 2:]  # (N, C, H, W/2)

        # Take mean across width dimension for each half
        left_mean = left.mean(dim=3, keepdim=True)  # (N, C, H, 1)
        right_mean = right.mean(dim=3, keepdim=True)  # (N, C, H, 1)

        # Concatenate along width dimension
        out = torch.cat([left_mean, right_mean], dim=3)  # (N, C, H, 2)
        return out

class GRADCAM_ResNet50_manual(nn.Module):
    def __init__(self, trained_model, hook_place):
        # hook_place: -3 means the next layer is -3, which is layer4
        # hook_place: -2 means the next layer is -2, which is pooling
        # hook_place: -1 means the next layer is -1, which is classifier

        super().__init__()

        # get the pretrained model
        self.m = trained_model
        self.hook_place = hook_place

        # get feature extraction layers
        self.features_conv = nn.Sequential(*list(self.m.children()))[:hook_place]

        # get layer 1-4
        self.layer1 = nn.Sequential(*list(self.m.children()))[-6]
        self.layer2 = nn.Sequential(*list(self.m.children()))[-5]
        self.layer3 = nn.Sequential(*list(self.m.children()))[-4]
        self.layer4 = nn.Sequential(*list(self.m.children()))[-3]

        # get the max pool
        self.max_pool = nn.Sequential(*list(self.m.children()))[-2]

        # get the classifier
        self.classifier = nn.Sequential(*list(self.m.children()))[-1]

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):

        x = self.features_conv(x)
        x.register_hook(self.activations_hook)
        if self.hook_place == -6:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.max_pool(x)
        elif self.hook_place == -5:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.max_pool(x)
        elif self.hook_place == -4:
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.max_pool(x)
        elif self.hook_place == -3:
            x = self.layer4(x)
            x = self.max_pool(x)
        elif self.hook_place == -2:
            x = self.max_pool(x)
        elif self.hook_place == -1:
            pass
        x = x.view((1, -1))  # reshapes into a 2D tensor of shape (1, N), where N is the total number of elements in x
        x = self.classifier(x)

        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_feature_maps(self, x):
        return self.features_conv(x)




class ResNet50_LRAP(ResNet):
    def __init__(self, num_classes=2):
        # Initialize a standard ResNet-50 skeleton
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])

        self.avgpool = LeftRightAvgPool()

        self.num_classes = num_classes

        self.fc = nn.Linear(2048 * 2, num_classes)

class ResNet50_Stem32_L1_64_L2_64_L3_64_rLRAP(ResNet):
    def __init__(self, num_classes=2):
        # Initialize a standard ResNet-50 skeleton
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])

        # --- Stem: change to 32 channels ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # --- Rebuild layer1 & layer2 using *self*._make_layer so self.inplanes is honored ---
        self.inplanes = 32  # <- IMPORTANT: set current width before building layer1
        self.layer1 = self._make_layer(Bottleneck, planes=16, blocks=3, stride=1)  # 16*4 = 64 out
        self.layer2 = self._make_layer(Bottleneck, planes=16, blocks=4, stride=1)  # 16*4 = 64 out
        self.layer3 = self._make_layer(Bottleneck, planes=16, blocks=4, stride=2)  # 16*4 = 64 out

        self.layer4 = nn.Identity()
        self.avgpool = LeftRightRowAvgPool()

        self.num_classes = num_classes

        # use the hardcoding option for classifier b/c otherwise fc does not appear in children()

        # --- Classifier (Option A: fixed input size, e.g., 224x224) ---
        self.fc = nn.Linear(64 * 53 * 2, num_classes)
        self._fc_built = True

        # --- Classifier (Option B: resolution-agnostic without LazyLinear) ---
        # Build fc dynamically on the first forward, based on actual feature size.

    #         self.fc = None
    #         self._fc_built = False

    def _build_fc_if_needed(self, x, num_classes):
        if not self._fc_built:
            in_feats = x.shape[1]  # flattened feature dimension
            self.fc = nn.Linear(in_feats, num_classes).to(x.device)
            self._fc_built = True

    def forward(self, x):
        # Stem
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x);
        x = self.maxpool(x)
        # Stages
        x = self.layer1(x)  # -> 64 ch
        x = self.layer2(x)  # -> 128 ch
        x = self.layer3(x)  # -> 128 ch
        x = self.avgpool(x)  # (left/right avg)
        # No layer3/layer4; keep spatial map
        x = torch.flatten(x, 1)  # (N, C*H*W)

        # If using Option B (dynamic fc):
        if self.fc is None or not self._fc_built:
            self._build_fc_if_needed(x, num_classes=self.num_classes)
        x = self.fc(x)
        return x

class ResNet50_Stem32_L1_64_L2_64_rLRAP(ResNet):
    def __init__(self, num_classes=2):
        # Initialize a standard ResNet-50 skeleton
        super().__init__(block=Bottleneck, layers=[3,4,6,3])

        # --- Stem: change to 32 channels ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        # --- Rebuild layer1 & layer2 using *self*._make_layer so self.inplanes is honored ---
        self.inplanes = 32                       # <- IMPORTANT: set current width before building layer1
        self.layer1 = self._make_layer(Bottleneck, planes=16, blocks=3, stride=1)  # 16*4 = 64 out
        self.layer2 = self._make_layer(Bottleneck, planes=16, blocks=4, stride=2)  # 16*4 = 64 out

        # --- Remove layer3, layer4, and GAP ---
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()
        self.avgpool = LeftRightRowAvgPool()

        self.num_classes = num_classes

        self.fc = nn.Linear(64 * 53 * 2, num_classes)
        self._fc_built = True

        # --- Classifier (Option B: resolution-agnostic without LazyLinear) ---
        # Build fc dynamically on the first forward, based on actual feature size.
        # self.fc = None
        # self._fc_built = False

    def _build_fc_if_needed(self, x, num_classes):
        if not self._fc_built:
            in_feats = x.shape[1]  # flattened feature dimension
            self.fc = nn.Linear(in_feats, num_classes).to(x.device)
            self._fc_built = True

    def forward(self, x):
        # Stem
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        # print(x.shape)
        # No layer3/layer4; keep spatial map
        x = torch.flatten(x, 1)     # (N, C*H*W)

        # If using Option B (dynamic fc):
        if self.fc is None or not self._fc_built:
            self._build_fc_if_needed(x, num_classes=self.num_classes)
        x = self.fc(x)
        return x

class ResNet50_Stem32_L1_64_rLRAP(ResNet):
    def __init__(self, num_classes=2):
        # Initialize a standard ResNet-50 skeleton
        super().__init__(block=Bottleneck, layers=[3,4,6,3])

        # --- Stem: change to 32 channels ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        # Use stride 2 in layer 1 to reduce spatial size to the desired level
        self.inplanes = 32                       # <- IMPORTANT: set current width before building layer1
        self.layer1 = self._make_layer(Bottleneck, planes=16, blocks=3, stride=2)  # 16*4 = 64 out

        # Remove layer2, layer3, layer4
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()

        # Use avgpool that averages left/right halves separately
        self.avgpool = LeftRightRowAvgPool()

        self.num_classes = num_classes

        # --- Classifier (Option A: fixed input size, e.g., 224x224) ---
        self.fc = nn.Linear(64 * 53 * 2, num_classes)
        self._fc_built = True

        # --- Classifier (Option B: resolution-agnostic without LazyLinear) ---
        # Build fc dynamically on the first forward, based on actual feature size.
        # self.fc = None
        # self._fc_built = False

    def _build_fc_if_needed(self, x, num_classes):
        if not self._fc_built:
            in_feats = x.shape[1]  # flattened feature dimension
            self.fc = nn.Linear(in_feats, num_classes).to(x.device)
            self._fc_built = True

    def forward(self, x):
        # Stem
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        # Stages
        x = self.layer1(x)          # -> 64 ch
        x = self.layer2(x)          # -> 128 ch
        x = self.avgpool(x)         # (left/right avg)
        # print(x.shape)
        # No layer3/layer4; keep spatial map
        x = torch.flatten(x, 1)     # (N, C*H*W)

        # If using Option B (dynamic fc):
        if self.fc is None or not self._fc_built:
            self._build_fc_if_needed(x, num_classes=self.num_classes)
        x = self.fc(x)
        return x




