"""
Grad-CAM explainability — Milestone 1 §"explainable predictions".

Produces a heatmap over the input leaf image showing which region drove
the model's prediction — lets a domain expert (or a farmer) sanity-check
that the model is looking at the lesion, not e.g. background soil or a
watermark (a documented failure mode for internet-scraped datasets like
PlantDoc, where some images may contain overlays/text).

Requires torch/torchvision. Not runnable in this research sandbox.
"""
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GradCAM:
    def __init__(self, model, target_layer):
        if not TORCH_AVAILABLE:
            raise ImportError("torch not installed in this environment")
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx
