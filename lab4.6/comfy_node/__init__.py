"""ComfyUI WAN LoRA Sampler — example custom node for lab 4.4.

After cloning this directory into `ComfyUI/custom_nodes/`, restart ComfyUI;
a "WAN LoRA Sampler" node appears under the "WAN" category.

The two exports below are how ComfyUI discovers a node package:
    NODE_CLASS_MAPPINGS         { id: class }      registers the node class
    NODE_DISPLAY_NAME_MAPPINGS  { id: label }      what shows in the UI
"""

from .nodes import WanLoraSampler

NODE_CLASS_MAPPINGS = {
    "WanLoraSampler": WanLoraSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLoraSampler": "WAN LoRA Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
