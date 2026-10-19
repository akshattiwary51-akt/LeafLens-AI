import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from explainability.gradcam import GradCAM


def _model(c=4):
    torch.manual_seed(0)
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, c)
    return m.eval()


def test_gradcam_shape_range_target_and_no_side_effects():
    m = _model()
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        before = m(x).clone()
    with GradCAM(m, m.features[-1]) as g:
        cam, idx = g.generate(x)
        cam2, _ = g.generate(x, class_idx=(idx + 1) % 4)
    assert cam.shape == (224, 224) and cam.dtype == np.float32
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert idx == int(before.argmax())
    assert cam2.shape == (224, 224)
    assert not m.training
    with torch.inference_mode():
        after = m(x)
    assert torch.equal(before, after)  # explanation did not change prediction
    assert len(m.features[-1]._forward_hooks) == 0  # hook cleaned up


def test_gradcam_target_class_changes_map():
    m = _model()
    x = torch.randn(1, 3, 224, 224)
    with GradCAM(m, m.features[-1]) as g:
        a, _ = g.generate(x, class_idx=0)
        b, _ = g.generate(x, class_idx=1)
    assert not np.allclose(a, b)
