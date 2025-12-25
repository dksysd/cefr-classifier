import torch
import torch.nn.functional as F

class OrdinalLabelSmoothingLoss(torch.nn.Module):
    """
    Ordinal Label Smoothing을 적용하는 손실 함수.
    정답 레이블의 (1-epsilon) 확률을 유지하고,
    인접한 좌/우 레이블에 각각 (epsilon/2) 만큼 확률을 배분합니다.
    """
    def __init__(self, alpha, epsilon=0.1, num_classes=6):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.alpha = alpha
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets):
        # 1. log_softmax 계산
        log_probs = self.log_softmax(inputs)

        # 2. OLS 타겟 분포 생성 (Vectorized)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            eps = self.epsilon
            half_eps = eps / 2.0

            # (1) 정답 레이블에 (1 - epsilon) 할당
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - eps)

            # (2) 왼쪽 인접 레이블 (target - 1)에 half_eps 할당
            left_idx = (targets - 1)
            left_mask = left_idx >= 0  # 0번 레이블(A1)은 왼쪽이 없음
            true_dist[left_mask, left_idx[left_mask]] += half_eps

            # (3) 오른쪽 인접 레이블 (target + 1)에 half_eps 할당
            right_idx = (targets + 1)
            right_mask = right_idx < self.num_classes # 5번(C2)은 오른쪽이 없음
            true_dist[right_mask, right_idx[right_mask]] += half_eps

            # (4) 엣지 케이스 보정 (0번과 5번)
            # 0번(A1)은 왼쪽에 줘야 할 half_eps가 갈 곳이 없으므로, 자신에게 다시 더함
            edge_0_mask = (targets == 0)
            true_dist[edge_0_mask, 0] += half_eps # (1-eps) + half_eps = 1 - half_eps

            # 5번(C2)은 오른쪽에 줘야 할 half_eps가 갈 곳이 없으므로, 자신에게 더함
            edge_5_mask = (targets == self.num_classes - 1)
            true_dist[edge_5_mask, self.num_classes - 1] += half_eps

            # (참고) A1(0) 타겟: [1-half_eps, half_eps, 0, 0, 0, 0]
            # (참고) C2(5) 타겟: [0, 0, 0, 0, half_eps, 1-half_eps]

        # 3. 손실 계산 (Cross-Entropy = -sum(target * log_prob))
        loss = -(true_dist * log_probs).sum(dim=-1)

        # 4. 클래스 가중치 적용 (FocalLoss의 alpha 기능)
        if self.alpha is not None:
            device = targets.device
            weights = self.alpha.to(device)[targets]
            loss = loss * weights

        return loss.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        weight = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss
