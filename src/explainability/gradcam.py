"""Grad-CAM for the PyTorch MobileNetV3-Small model (research/server side).

Grad-CAM needs gradients and intermediate activations, so it runs on the
PyTorch model only. It is NOT available in TFLite/ONNX runtime inference;
an on-device alternative would have to be a separate, clearly-labelled
method (e.g. occlusion sensitivity), not Grad-CAM.

Implementation notes: activations/gradients are captured with a forward hook
plus a tensor hook on the layer output (module backward hooks break with
MobileNetV3's in-place Hardswish). Hooks are removed by ``close()``/context
manager so they do not leak or alter later predictions. The CAM is computed
on the *raw logit* of the target class.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "gradients", g.detach()))

    def close(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def generate(self, input_tensor, class_idx=None):
        """Return (cam HxW float32 in [0,1] at input resolution, class_idx)."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                x = input_tensor.detach().clone().requires_grad_(True)
                self.gradients = None
                out = self.model(x)
                if class_idx is None:
                    class_idx = int(out.argmax(dim=1).item())
                self.model.zero_grad(set_to_none=True)
                out[0, class_idx].backward()
            if self.gradients is None:
                raise RuntimeError("no gradients captured at target layer")
            w = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                                mode="bilinear", align_corners=False)
            cam = cam[0, 0].cpu().numpy().astype(np.float32)
            rng = cam.max() - cam.min()
            cam = (cam - cam.min()) / rng if rng > 1e-12 else np.zeros_like(cam)
            return cam, class_idx
        finally:
            self.model.train(was_training)
