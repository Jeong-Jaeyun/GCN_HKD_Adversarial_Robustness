
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistillationLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        task_loss = self.ce_loss(student_logits, targets)


        student_log_probs = F.log_softmax(
            student_logits / self.temperature,
            dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature,
            dim=-1
        )


        kd_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)


        combined_loss = self.alpha * kd_loss + (1 - self.alpha) * task_loss

        return combined_loss


class HierarchicalDistillationLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 4.0,
        feature_alpha: float = 0.3,
        logit_alpha: float = 0.5,
        task_alpha: float = 0.2,
        layer_weights: Optional[dict] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.feature_alpha = feature_alpha
        self.logit_alpha = logit_alpha
        self.task_alpha = task_alpha
        self.reduction = reduction


        self.layer_weights = layer_weights or {}


        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[dict] = None,
        teacher_features: Optional[dict] = None
    ) -> torch.Tensor:
        losses = {}


        task_loss = self.ce_loss(student_logits, targets)
        losses['task'] = task_loss


        student_probs = F.log_softmax(
            student_logits / self.temperature,
            dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature,
            dim=-1
        )
        logit_loss = self.kl_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        losses['logit_distillation'] = logit_loss


        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
            num_matched_layers = 0

            for layer_idx in student_features.keys():
                if layer_idx in teacher_features:
                    s_feat = student_features[layer_idx]
                    t_feat = teacher_features[layer_idx]


                    if s_feat.shape != t_feat.shape:

                        if s_feat.shape[1] < t_feat.shape[1]:
                            t_feat = self._project_features(t_feat, s_feat.shape[1])
                        elif s_feat.shape[1] > t_feat.shape[1]:
                            s_feat = self._project_features(s_feat, t_feat.shape[1])


                    layer_loss = self.mse_loss(s_feat, t_feat)


                    if layer_idx in self.layer_weights:
                        layer_loss = layer_loss * self.layer_weights[layer_idx]

                    feature_loss += layer_loss
                    num_matched_layers += 1

            if num_matched_layers > 0:
                feature_loss = feature_loss / num_matched_layers

        losses['feature_distillation'] = feature_loss


        combined_loss = (
            self.task_alpha * task_loss +
            self.logit_alpha * logit_loss +
            self.feature_alpha * feature_loss
        )

        return combined_loss, losses

    def _project_features(self, features: torch.Tensor, target_dim: int) -> torch.Tensor:
        if features.shape[1] == target_dim:
            return features


        if features.shape[1] > target_dim:

            kernel_size = features.shape[1] // target_dim
            features = F.avg_pool1d(
                features.unsqueeze(0),
                kernel_size=kernel_size,
                stride=kernel_size
            ).squeeze(0)
        else:

            repeat_factor = (target_dim + features.shape[1] - 1) // features.shape[1]
            features = features.repeat_interleave(repeat_factor, dim=1)
            features = features[:, :target_dim]

        return features


class RobustnessLoss(nn.Module):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(
        self,
        clean_logits: torch.Tensor,
        perturbed_logits: torch.Tensor
    ) -> torch.Tensor:
        clean_probs = F.log_softmax(clean_logits, dim=-1)
        perturbed_probs = F.softmax(perturbed_logits, dim=-1)

        loss = self.kl_loss(clean_probs, perturbed_probs)

        return loss


class CombinedLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 4.0,
        task_weight: float = 1.0,
        kd_weight: float = 0.5,
        robustness_weight: float = 0.3
    ):
        super().__init__()
        self.task_weight = task_weight
        self.kd_weight = kd_weight
        self.robustness_weight = robustness_weight

        self.distillation_loss = DistillationLoss(temperature=temperature)
        self.robustness_loss = RobustnessLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        clean_logits: Optional[torch.Tensor] = None,
        perturbed_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        kd_loss = self.distillation_loss(student_logits, teacher_logits, targets)


        robustness_loss = 0.0
        if clean_logits is not None and perturbed_logits is not None:
            robustness_loss = self.robustness_loss(clean_logits, perturbed_logits)


        combined_loss = (
            self.task_weight * kd_loss +
            self.kd_weight * kd_loss +
            self.robustness_weight * robustness_loss
        )

        return combined_loss
