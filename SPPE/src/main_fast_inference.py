import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from SPPE.src.models.FastPose import FastPose

import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class InferenNet_fastRes50(nn.Module):
    def __init__(self, weights_file='Models/fast_res50_256x192.pth'):
        super().__init__()

        self.pyranet = FastPose('resnet50', 17).cuda()
        print('Loading pose model from {}'.format(weights_file))
        self.pyranet.load_state_dict(torch.load(weights_file))
        self.pyranet.eval()

    def forward(self, x):
        out = self.pyranet(x)

        return out
