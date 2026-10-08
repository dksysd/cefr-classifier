import pytest
import torch

from training.loss import OrdinalLabelSmoothingLoss


def target_distribution(targets, epsilon=0.1):
    """loss = -sum(true_dist * log_probs) 이므로 logits=0 에서의 gradient 로 true_dist 를 복원"""
    logits = torch.zeros(len(targets), 6, requires_grad=True)
    loss = OrdinalLabelSmoothingLoss(alpha=None, epsilon=epsilon)(logits, torch.tensor(targets))
    loss.backward()
    # d(loss)/d(logits) = (softmax - true_dist) / batch
    return torch.full((len(targets), 6), 1 / 6) - logits.grad * len(targets)


def test_target_distribution_middle_and_edges():
    dist = target_distribution([0, 2, 5])
    expected = torch.tensor([
        [0.95, 0.05, 0, 0, 0, 0],
        [0, 0.05, 0.9, 0.05, 0, 0],
        [0, 0, 0, 0, 0.05, 0.95],
    ])
    assert torch.allclose(dist, expected, atol=1e-6)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_class_weights_scale_loss():
    logits = torch.randn(4, 6)
    targets = torch.tensor([0, 1, 2, 3])
    base = OrdinalLabelSmoothingLoss(alpha=None)(logits, targets)
    doubled = OrdinalLabelSmoothingLoss(alpha=torch.full((6,), 2.0))(logits, targets)
    assert doubled.item() == pytest.approx(2 * base.item(), rel=1e-5)


def test_closer_prediction_has_lower_loss():
    loss_fn = OrdinalLabelSmoothingLoss(alpha=None)
    target = torch.tensor([2])
    near = torch.tensor([[0.0, 0, 0, 5, 0, 0]])  # B2 로 예측 (인접)
    far = torch.tensor([[0.0, 0, 0, 0, 0, 5]])   # C2 로 예측 (먼 레벨)
    assert loss_fn(near, target) < loss_fn(far, target)
