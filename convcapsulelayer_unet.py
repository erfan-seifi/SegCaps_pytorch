import torch
import torch.nn as nn
import torch.nn.functional as F


def _squash_2(input_tensor):
    norm = torch.norm(input_tensor, dim=3, keepdim=True)
    norm_squared = norm * norm + 1e-4
    return (norm / (1 + norm_squared)) * input_tensor


class ConvCapsuleLayer(nn.Module):
    def __init__(self, kernel_size, NP, LC, LP, routings, strides=1, padding='same', deconv=False, simple=False):
        super(ConvCapsuleLayer, self).__init__()
        self.kernel_size = kernel_size
        self.NP = NP
        self.LP = LP
        if strides != 1:
            self.padding1 = (kernel_size + 1) // 2 - 1
            self.padding2 = self.padding1 #kernel_size - self.padding1 
        else:
            self.padding1 = (kernel_size - 1) // 2 
            self.padding2 = self.padding1
        self.strides = strides
        self.padding3 = padding
        self.routing_nums = routings
        self.LC = LC
        self.deconv = deconv
        self.simple = simple

        if not self.deconv:
            self.conv2d_for_cap = nn.Conv2d(in_channels=self.LC,
                                            out_channels=self.NP * self.LP,
                                            kernel_size=self.kernel_size,
                                            padding=(self.padding1, self.padding2),
                                            stride=strides,
                                            bias=False)
        if self.deconv:
            self.conv2d_for_cap = nn.ConvTranspose2d(in_channels=self.LC,
                                            out_channels=self.NP * self.LP,
                                            kernel_size=self.kernel_size,
                                            padding=(self.padding1, self.padding2),
                                            #output_padding=(1, 1),
                                            stride=2,
                                            bias=False) 
        if self.simple:
            self.conv2d_for_cap = nn.Conv2d(in_channels=self.LC,
                                            out_channels=self.NP * self.LP,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding3,
                                            bias=True)

        self.conv2d_for_cap.weight.data.uniform_(0.0, 0.1)
        print("uniform [0, 0.1] for caps conv weights")

    def forward(self, x):
        # NC: number of child capsules
        # LC: child capsule length
        # NP: number of parent capsules
        B, NC, LC, H, W = x.shape
        LP = self.LP
        NP = self.NP
        
        x = x.view(B * NC, LC, H, W)
        # (B * NC), (NP * LP), H, W
        votes = self.conv2d_for_cap(x)
        _, _, H, W = votes.shape
        votes = votes.view(B, NC, NP, LP, H, W)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # sims = torch.zeros(B, NC, NP, 1, H, W).to(device)
        sims = torch.zeros(B, NC, NP, 1, H, W)
        sims = sims.cuda(x.get_device()) if x.is_cuda else sims
        for iterations in range(self.routing_nums):
            # c_ij: B, NC, NP, 1, H, W
            c_ij = F.softmax(sims + 1e-4, dim=2)
            # sum(v_ij * c_ij): B, NC, NP, LP, H, W -> B, 1, NP, LP, H, W
            parent_beforeSquash = torch.sum(c_ij * votes, dim=1, keepdim=True)
            parent = _squash_2(parent_beforeSquash)
            sims += torch.sum(votes * parent, dim=3, keepdim=True)
        
        # B, 1, NP, LP, H, W -> B, NP, LP, H, W
        parent = parent.squeeze(dim=1)
        return parent
