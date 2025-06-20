import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init_old(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

def initial_model_weight_old(layers):

    for layer in layers:
        if list(layer.children()) == []:
            weights_init_old(layer)
            # print('weight initial finished!')
        else:
            for sub_layer in list(layer.children()):
                initial_model_weight_old([sub_layer])
                
def initial_model_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)        




class HCN(nn.Module):
    """Backbone of Co-occurrence Feature Learning from Skeleton Data for Action
    Recognition and Detection with Hierarchical Aggregation.

    Args:
        in_channels (int): Number of channels in the input data.
        num_joints (int): Number of joints in each skeleton.
        clip_len (int): Skeleton sequence length.
        with_bn (bool): Whether to append a BN layer after conv1.
        reduce (str): Reduction mode along the temporal dimension,'flatten' or
            'mean'.
        pretrained (str | None): Name of pretrained model.

    Shape:
        - Input: :math:`(N, in_channels, T, V, M)`
        - Output: :math:`(N, D)` where
            :math:`N` is a batch size,
            :math:`T` is a length of input sequence,
            :math:`V` is the number of joints,
            :math:`M` is the number of instances in a frame.
    """

    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=1,
                 out_channel=64,
                 window_size=64,
                 num_class = 60):
        super().__init__()
        self.num_person = num_person

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel//2, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.5))
        
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=out_channel//2, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.5))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )


        self.fc7= nn.Sequential(
            nn.Linear(256 * 2 * window_size // 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.fc8 = nn.Linear(256,num_class)
        #initial_model_weight(self)
        initial_model_weight_old(layers = list(self.children()))
        

    def forward(self, x,target=None):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)


        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])

            out = self.conv2(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            out = self.conv3(out)
            out_p = self.conv4(out)


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            out = self.conv2m(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            out = self.conv5(out)
            out = self.conv6(out)

            logits.append(out)

        # max out logits
        #out = torch.max(logits[0],logits[1])
        out = logits[0]
        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out
    




