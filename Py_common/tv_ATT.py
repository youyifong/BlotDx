import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



class ResNet50_CBAM(nn.Module):
    def __init__(self, 
                 nchan = 3, 
                 num_classes = 2, 
                 pretrained_model = 'Resnet50_withPretrainedWeight', 
                 use_cbam_class = False,
                 reduction_ratio = 1,
                 kernel_cbam = 3, 
                 freeze = False):
        super(ResNet50_CBAM, self).__init__()
        
        #Get Parameters
        self.use_cbam_class = use_cbam_class

        # Load a pretrained model
        if pretrained_model == 'Resnet50':
            m = models.resnet50()
        elif pretrained_model == "Resnet50_withPretrainedWeight":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        #Get feature dim for fc
        fc_dim = m.fc.in_features

        #Modify the first convolutional layer to accept nchan input channels
        m.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False) # Train1
        
        #Remove classifaction layer from the original pretrained model
        self.features = nn.Sequential(*list(m.children())[:-2])

        #define self.cbam
        if self.use_cbam_class:
            self.cbam = CBAM(n_channels_in = fc_dim, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)

        #classification layer
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(fc_dim, num_classes)

        #Freeze the layers before the last layer
        if freeze:
            for name, param in self.m.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False


    def forward(self, x):
        out = self.features(x)

        if self.use_cbam_class:
            cban_out, chan_att, spat_att = self.cbam(out)
            out = out  + cban_out
        
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out, chan_att, spat_att

class CBAM(nn.Module):

    def __init__(self, n_channels_in = 3, reduction_ratio = 2, kernel_size = 3):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp, chan_att, spat_att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out





#GAIN model
class GAIN_V2(nn.Module):
    def __init__(self, 
                 nchan=3, 
                 num_classes=2,
                 sigma=0.5,
                 omega=10,
                 hook_place=-2):
        super(GAIN_V2, self).__init__()
        
        self.sigma = sigma
        self.omega = omega
        
        # Load a pretrained model
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        fc_dim = m.fc.in_features
        
        #Modify the 1st conv
        m.conv1 = nn.Conv2d(nchan, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        #Grab feature layers
        self.features_conv = nn.Sequential(*list(m.children())[:hook_place])
        
        #Pooling layer
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        
        #Classification layer
        self.classifier = nn.Linear(fc_dim, num_classes)

    def forward(self, x, labels):
        
        #Get original image
        I = x.clone()
        
        
        # Pass 1: Grad-CAM weights (detached constants)
        feat = self.features_conv(x) #torch.Size([1, 2048, 14, 2])
        feat.requires_grad_(True)
        pooled = self.avgpool1(feat)
        flat = torch.flatten(pooled, 1)
        out = self.classifier(flat)
        targets = out.argmax(1) if labels is None else labels
        score = out[torch.arange(out.size(0), device=out.device), targets].sum()
        grads = torch.autograd.grad(score, feat, retain_graph=True, create_graph=False)[0]
        weights = grads.mean(dim=(2,3), keepdim=True).relu().detach() #torch.Size([1, 2048, 1, 1])
        
        #weights  = self.compute_cam_weights(x, labels)
        #pass 2: training graph
        #Get feature maps
        #x = self.features_conv(x)
        
        # #Classifition
        # pooled_x = self.avgpool1(x)
        # flat_x = torch.flatten(pooled_x, 1)
        # out = self.classifier(flat_x)
        
        #Get attention map (equation 2), 
        A_c = self. get_attention_map(feat,weights)
        A_c_up = F.interpolate(A_c, size=I.shape[2:], mode='bilinear', align_corners=False) #Upsampled A_c
        A_c_scaled = self.minmax01(A_c_up)  #scaled


        #Get I_star (region beyound the network's current attention)
        I_star = self.generate_masked_input(I, A_c_scaled)
        
        #Prediction using I_star
        out_star = self.features_conv(I_star)
        out_star = self.avgpool1(out_star)
        out_star = torch.flatten(out_star, 1)
        out_star = self.classifier(out_star)
            
        return out, out_star, A_c ,weights

    def threshold_mask(self, att_map, omega, sigma): #checked
        return torch.sigmoid(omega * (att_map - sigma))

    def generate_masked_input(self, input_image, attention_map): #checked
        #equation 3
        T = self.threshold_mask(attention_map, self.omega, self.sigma) #apply threshold
        I_c = input_image - (T * input_image)
        return I_c
    
    def get_feature_maps(self, input_image):
        return self.features_conv(input_image)
    
    def get_attention_map(self, fl, wc):

        Ac = torch.mul(fl, wc).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)
        
        #for each sample, using its own kernel weights[i]
        #equivalent to:
        # Ac = torch.cat([
        #     F.relu(F.conv2d(fl[i:i+1], wc[i].unsqueeze(0)))
        #     for i in range(fl.size(0))
        # ], dim=0)
        
        return Ac


    def compute_cam_weights(self, input_image, labels=None):
        
        #Get feature
        fs = self.get_feature_maps(input_image)    
        #require gradient             
        fs.requires_grad_(True)
        
        #Classifition
        pooled = self.avgpool1(fs)
        flat = torch.flatten(pooled, 1)
        pred_logits = self.classifier(flat) #[B, C]
        
        if labels is None:
            targets = pred_logits.argmax(dim=1)    # [B] #for prediction, using the predicted class
        else:
            targets = labels
            
        #Get Predicted logits on the target class
        pred_logits = pred_logits[torch.arange(pred_logits.size(0), device=pred_logits.device), targets].sum() #sum over batches just make it a scalar for autograd, 
        
        #Compute gradiant wrt to pred_logits 
        grads = torch.autograd.grad(pred_logits, fs, retain_graph=False, create_graph=False)[0]

        w = grads.mean(dim=(2, 3), keepdim=True).detach()     
        
        return w
    
    def minmax01(self, x, eps=1e-8):
        # Normalize PER-IMAGE, not across the batch
        x_min = x.amin(dim=(2,3), keepdim=True)
        x_max = x.amax(dim=(2,3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    
