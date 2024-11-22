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

STRENGTHS = ["very high", "high", "medium", "low", "very low"]
STRENGTHS_VALUES = [1,2, 3,4,5]

class StyleModelApplySimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "image_strength": (STRENGTHS, {"default": "medium"})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision_output, style_model, conditioning, image_strength):
        stren = STRENGTHS.index(image_strength)
        downsampling_factor = STRENGTHS_VALUES[stren]
        mode="area"
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if downsampling_factor>1:
            (b,t,h)=cond.shape
            m = int(np.sqrt(t))
            cond=torch.nn.functional.interpolate(cond.view(b, m, m, h).transpose(1,-1), size=(m//downsampling_factor, m//downsampling_factor), mode=mode)#
            cond=cond.transpose(1,-1).reshape(b,-1,h)
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )


class StyleModelApplyInterpolation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "downsampling_factor": ("INT", {"default": 3, "min": 1, "max": 9}),
                             "mode": (["nearest", "bicubic", "bilinear","area", "nearest-exact" ], {"default": "area"})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, clip_vision_output, style_model, conditioning, downsampling_factor, mode):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if downsampling_factor>1:
            (b,t,h)=cond.shape
            m = int(np.sqrt(t))
            cond=torch.nn.functional.interpolate(cond.view(b, m, m, h).transpose(1,-1), size=(m//downsampling_factor, m//downsampling_factor), mode=mode)#
            cond=cond.transpose(1,-1).reshape(b,-1,h)
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )


class StyleModelApplyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "downsampling_factor": ("INT", {"default": 3, "min": 1, "max": 9}),
                             "mode": (["nearest", "bicubic", "bilinear","area", "nearest-exact" ], {"default": "area"}),
                             "merge_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "clipWeight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "_for_testing/conditioning/style_model"

    def apply_stylemodel(self, clip_vision_output, style_model, conditioning, downsampling_factor, mode, merge_strength, clipWeight):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if downsampling_factor>1:
            (b,t,h)=cond.shape
            m = int(np.sqrt(t))
            cond=torch.nn.functional.interpolate(cond.view(b, m, m, h).transpose(1,-1), size=(m//downsampling_factor, m//downsampling_factor), mode=mode)#
            cond=cond.transpose(1,-1).reshape(b,-1,h)

        cond = automerge(cond, merge_strength)*clipWeight
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )




# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "StyleModelApplySimple": StyleModelApplySimple,
    "StyleModelApplyAdvanced": StyleModelApplyAdvanced,
    "StyleModelApplyInterpolation": StyleModelApplyInterpolation
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelApplySimple": "Apply style model (simple)",
    "StyleModelApplyAdvanced": "Apply Style model (advanced)",
    "StyleModelApplyInterpolation": "Apply style model (advanced)"
}