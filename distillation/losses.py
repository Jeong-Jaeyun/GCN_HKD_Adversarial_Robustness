import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def kd_term(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        task_loss = self.ce_loss(student_logits, targets)
        kd_loss = self.kd_term(student_logits, teacher_logits)
        return self.alpha * kd_loss + (1.0 - self.alpha) * task_loss


class HierarchicalDistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        feature_alpha: float = 0.3,
        logit_alpha: float = 0.5,
        task_alpha: float = 1.0,
        consistency_alpha: float = 0.0,
        layer_weights: Optional[Dict[int, float]] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.feature_alpha = feature_alpha
        self.logit_alpha = logit_alpha
        self.task_alpha = task_alpha
        self.consistency_alpha = consistency_alpha
        self.layer_weights = layer_weights or {}

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()
        self.feature_projectors = nn.ModuleDict()

    def _kd_term(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)

    def _consistency_term(
        self,
        clean_logits: torch.Tensor,
        perturbed_logits: torch.Tensor,
    ) -> torch.Tensor:
        clean_log_probs = F.log_softmax(clean_logits, dim=-1)
        perturbed_probs = F.softmax(perturbed_logits.detach(), dim=-1)
        perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)
        clean_probs = F.softmax(clean_logits.detach(), dim=-1)
        loss_a = self.kl_loss(clean_log_probs, perturbed_probs)
        loss_b = self.kl_loss(perturbed_log_probs, clean_probs)
        return 0.5 * (loss_a + loss_b)

    def _to_graph_embedding(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() == 1:
            return feature.unsqueeze(0)
        if feature.dim() == 2:
            if feature.size(0) == 1:
                return feature
            return feature.mean(dim=0, keepdim=True)
        flattened = feature.reshape(feature.size(0), -1)
        return flattened.mean(dim=0, keepdim=True)

    def _project_teacher_feature(
        self,
        teacher_feature: torch.Tensor,
        student_dim: int,
        layer_key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        teacher_dim = teacher_feature.size(-1)
        if teacher_dim == student_dim:
            return teacher_feature

        projector_key = f"{layer_key}_{teacher_dim}_{student_dim}"
        if projector_key not in self.feature_projectors:
            projector = nn.Sequential(
                nn.Linear(teacher_dim, student_dim),
                nn.GELU(),
                nn.Linear(student_dim, student_dim),
            ).to(device=device, dtype=dtype)
            self.feature_projectors[projector_key] = projector

        projector = self.feature_projectors[projector_key]
        return projector(teacher_feature)

    def _feature_alignment_term(
        self,
        student_features: Optional[Dict[int, torch.Tensor]],
        teacher_features: Optional[Dict[int, torch.Tensor]],
        prefix: str,
        default_device: torch.device,
        default_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not student_features or not teacher_features:
            return torch.zeros((), device=default_device, dtype=default_dtype)

        common_layers = sorted(set(student_features.keys()).intersection(teacher_features.keys()))
        if not common_layers:
            return torch.zeros((), device=default_device, dtype=default_dtype)

        layer_losses = []
        for layer_idx in common_layers:
            student_feature = self._to_graph_embedding(student_features[layer_idx])
            teacher_feature = self._to_graph_embedding(teacher_features[layer_idx]).to(
                device=student_feature.device,
                dtype=student_feature.dtype,
            )
            teacher_feature = self._project_teacher_feature(
                teacher_feature,
                student_feature.size(-1),
                f"{prefix}_layer{layer_idx}",
                student_feature.device,
                student_feature.dtype,
            )
            layer_loss = self.mse_loss(student_feature, teacher_feature)
            if layer_idx in self.layer_weights:
                layer_loss = layer_loss * float(self.layer_weights[layer_idx])
            layer_losses.append(layer_loss)

        if not layer_losses:
            return torch.zeros((), device=default_device, dtype=default_dtype)

        return torch.stack(layer_losses).mean()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[Dict[int, torch.Tensor]] = None,
        teacher_features: Optional[Dict[int, torch.Tensor]] = None,
    ):
        task_loss = self.ce_loss(student_logits, targets)
        logit_loss = self._kd_term(student_logits, teacher_logits)
        feature_loss = self._feature_alignment_term(
            student_features=student_features,
            teacher_features=teacher_features,
            prefix="clean",
            default_device=student_logits.device,
            default_dtype=student_logits.dtype,
        )

        total_loss = (
            self.task_alpha * task_loss
            + self.logit_alpha * logit_loss
            + self.feature_alpha * feature_loss
        )

        losses = {
            "task": task_loss,
            "logit_distillation": logit_loss,
            "feature_distillation": feature_loss,
            "consistency": torch.zeros((), device=student_logits.device, dtype=student_logits.dtype),
        }
        return total_loss, losses

    def forward_robust(
        self,
        student_clean_logits: torch.Tensor,
        student_perturbed_logits: torch.Tensor,
        teacher_clean_logits: torch.Tensor,
        targets: torch.Tensor,
        student_clean_features: Optional[Dict[int, torch.Tensor]] = None,
        student_perturbed_features: Optional[Dict[int, torch.Tensor]] = None,
        teacher_clean_features: Optional[Dict[int, torch.Tensor]] = None,
    ):
        task_clean = self.ce_loss(student_clean_logits, targets)
        task_perturbed = self.ce_loss(student_perturbed_logits, targets)
        task_loss = 0.5 * (task_clean + task_perturbed)

        logit_clean = self._kd_term(student_clean_logits, teacher_clean_logits)
        logit_perturbed = self._kd_term(student_perturbed_logits, teacher_clean_logits)
        logit_loss = 0.5 * (logit_clean + logit_perturbed)

        feature_clean = self._feature_alignment_term(
            student_features=student_clean_features,
            teacher_features=teacher_clean_features,
            prefix="clean",
            default_device=student_clean_logits.device,
            default_dtype=student_clean_logits.dtype,
        )
        feature_perturbed = self._feature_alignment_term(
            student_features=student_perturbed_features,
            teacher_features=teacher_clean_features,
            prefix="perturbed",
            default_device=student_clean_logits.device,
            default_dtype=student_clean_logits.dtype,
        )
        feature_loss = 0.5 * (feature_clean + feature_perturbed)

        consistency_loss = self._consistency_term(student_clean_logits, student_perturbed_logits)

        total_loss = (
            self.task_alpha * task_loss
            + self.logit_alpha * logit_loss
            + self.feature_alpha * feature_loss
            + self.consistency_alpha * consistency_loss
        )

        losses = {
            "task": task_loss,
            "task_clean": task_clean,
            "task_perturbed": task_perturbed,
            "logit_distillation": logit_loss,
            "logit_clean": logit_clean,
            "logit_perturbed": logit_perturbed,
            "feature_distillation": feature_loss,
            "feature_clean": feature_clean,
            "feature_perturbed": feature_perturbed,
            "consistency": consistency_loss,
        }
        return total_loss, losses


class RobustnessLoss(nn.Module):
    def __init__(self, reduction: str = "batchmean"):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(
        self,
        clean_logits: torch.Tensor,
        perturbed_logits: torch.Tensor,
    ) -> torch.Tensor:
        clean_log_probs = F.log_softmax(clean_logits, dim=-1)
        perturbed_probs = F.softmax(perturbed_logits, dim=-1)
        return self.kl_loss(clean_log_probs, perturbed_probs)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        task_weight: float = 1.0,
        kd_weight: float = 0.5,
        robustness_weight: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.task_weight = task_weight
        self.kd_weight = kd_weight
        self.robustness_weight = robustness_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.robustness_loss = RobustnessLoss()

    def _kd_term(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        clean_logits: Optional[torch.Tensor] = None,
        perturbed_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        task_loss = self.ce_loss(student_logits, targets)
        kd_loss = self._kd_term(student_logits, teacher_logits)

        robust_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        if clean_logits is not None and perturbed_logits is not None:
            robust_loss = self.robustness_loss(clean_logits, perturbed_logits)

        return (
            self.task_weight * task_loss
            + self.kd_weight * kd_loss
            + self.robustness_weight * robust_loss
        )
