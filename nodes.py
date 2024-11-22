import numpy as np
import torch
import comfy
import folder_paths
import nodes
import os
import math
import re
import safetensors
import glob
from collections import namedtuple

@torch.no_grad()
def automerge(tensor, threshold):
    (batchsize, slices, dim) = tensor.shape
    newTensor=[]
    for batch in range(batchsize):
        tokens = []
        lastEmbed = tensor[batch,0,:]
        merge=[lastEmbed]
        tokens.append(lastEmbed)
        for i in range(1,slices):
            tok = tensor[batch,i,:]
            cosine = torch.dot(tok,lastEmbed)/torch.sqrt(torch.dot(tok,tok)*torch.dot(lastEmbed,lastEmbed))
            if cosine >= threshold:
                merge.append(tok)
                lastEmbed = torch.stack(merge).mean(dim=0)
            else:
                tokens.append(lastEmbed)
                merge=[]
                lastEmbed=tok
        newTensor.append(torch.stack(tokens))
    return torch.stack(newTensor)

DownScalingFactors = {"1:1": 1, "1:3":3, "1:9": 9}

class StyleModelApplyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "downscale": (["1:1", "1:3", "1:9"], {"default": "1:1"}),
                             "merge_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "clipWeight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision_output, style_model, conditioning, downscale, merge_strength, clipWeight):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if downscale!="1:1":
            (b,t,h)=cond.shape
            m = int(np.sqrt(t))
            factor = DownScalingFactors[downscale]
            cond=torch.nn.PixelUnshuffle(factor)(cond.view(b, m, m, h).transpose(-1, 1)).transpose(1,-1)
            cond=cond.view(b,m//factor, m//factor, h, factor*factor).mean(-1).view(b,-1, h)

        cond = automerge(cond, merge_strength)*clipWeight
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )




# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "StyleModelApplyAdvanced": StyleModelApplyAdvanced,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelApplyAdvanced": "Apply Style model (advanced)"
}