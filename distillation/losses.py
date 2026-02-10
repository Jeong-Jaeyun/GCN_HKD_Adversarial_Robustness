"""Loss functions for knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistillationLoss(nn.Module):
    """Standard knowledge distillation loss.
    
    Combines task loss (CE) with distillation loss (KL divergence)
    between student and teacher logits.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = 'mean'
    ):
        """Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax scaling
            alpha: Weight for distillation loss (task loss weight = 1 - alpha)
            reduction: 'mean', 'sum', or 'none'
        """
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
        """Compute distillation loss.
        
        Args:
            student_logits: Student model logits (batch_size, num_classes)
            teacher_logits: Teacher model logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            combined_loss = alpha * KL_loss + (1 - alpha) * CE_loss
        """
        # Task loss (cross-entropy on student logits)
        task_loss = self.ce_loss(student_logits, targets)
        
        # Distillation loss (KL divergence between distributions at temperature T)
        student_log_probs = F.log_softmax(
            student_logits / self.temperature,
            dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature,
            dim=-1
        )
        
        # Scale KL loss by T^2 to account for temperature
        kd_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Combined loss
        combined_loss = self.alpha * kd_loss + (1 - self.alpha) * task_loss
        
        return combined_loss


class HierarchicalDistillationLoss(nn.Module):
    """Hierarchical knowledge distillation loss.
    
    Transfers knowledge from teacher to student at multiple levels:
    - Feature level (hidden layer representations)
    - Logit level (task predictions)
    
    This is key to Phase 3 of the architecture.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        feature_alpha: float = 0.3,
        logit_alpha: float = 0.5,
        task_alpha: float = 0.2,
        layer_weights: Optional[dict] = None,
        reduction: str = 'mean'
    ):
        """Initialize hierarchical distillation loss.
        
        Args:
            temperature: Temperature for softmax scaling
            feature_alpha: Weight for feature-level distillation
            logit_alpha: Weight for logit-level distillation
            task_alpha: Weight for task loss
            layer_weights: Dict mapping layer indices to loss weights
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.temperature = temperature
        self.feature_alpha = feature_alpha
        self.logit_alpha = logit_alpha
        self.task_alpha = task_alpha
        self.reduction = reduction
        
        # Layer-specific weights
        self.layer_weights = layer_weights or {}
        
        # Loss components
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
        """Compute hierarchical distillation loss.
        
        Args:
            student_logits: Student logits (batch_size, num_classes)
            teacher_logits: Teacher logits (batch_size, num_classes)
            targets: Ground truth labels
            student_features: Dict mapping layer idx to feature tensors
            teacher_features: Dict mapping layer idx to feature tensors
            
        Returns:
            Hierarchical distillation loss
        """
        losses = {}
        
        # Task loss
        task_loss = self.ce_loss(student_logits, targets)
        losses['task'] = task_loss
        
        # Logit-level distillation loss
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
        
        # Feature-level distillation loss
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
            num_matched_layers = 0
            
            for layer_idx in student_features.keys():
                if layer_idx in teacher_features:
                    s_feat = student_features[layer_idx]
                    t_feat = teacher_features[layer_idx]
                    
                    # Match dimensions if necessary
                    if s_feat.shape != t_feat.shape:
                        # Use linear projection or pooling to match dimensions
                        if s_feat.shape[1] < t_feat.shape[1]:
                            t_feat = self._project_features(t_feat, s_feat.shape[1])
                        elif s_feat.shape[1] > t_feat.shape[1]:
                            s_feat = self._project_features(s_feat, t_feat.shape[1])
                    
                    # Compute layer-specific loss
                    layer_loss = self.mse_loss(s_feat, t_feat)
                    
                    # Apply layer weight if specified
                    if layer_idx in self.layer_weights:
                        layer_loss = layer_loss * self.layer_weights[layer_idx]
                    
                    feature_loss += layer_loss
                    num_matched_layers += 1
            
            if num_matched_layers > 0:
                feature_loss = feature_loss / num_matched_layers
        
        losses['feature_distillation'] = feature_loss
        
        # Combined loss
        combined_loss = (
            self.task_alpha * task_loss +
            self.logit_alpha * logit_loss +
            self.feature_alpha * feature_loss
        )
        
        return combined_loss, losses
    
    def _project_features(self, features: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Project features to target dimension."""
        if features.shape[1] == target_dim:
            return features
        
        # Simple linear projection
        if features.shape[1] > target_dim:
            # Reduce dimension with average pooling
            kernel_size = features.shape[1] // target_dim
            features = F.avg_pool1d(
                features.unsqueeze(0),
                kernel_size=kernel_size,
                stride=kernel_size
            ).squeeze(0)
        else:
            # Increase dimension by repeating
            repeat_factor = (target_dim + features.shape[1] - 1) // features.shape[1]
            features = features.repeat_interleave(repeat_factor, dim=1)
            features = features[:, :target_dim]
        
        return features


class RobustnessLoss(nn.Module):
    """Loss for adversarial robustness training.
    
    Encourages the model to make consistent predictions on
    adversarially perturbed samples.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """Initialize robustness loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)
    
    def forward(
        self,
        clean_logits: torch.Tensor,
        perturbed_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute robustness loss.
        
        Args:
            clean_logits: Logits from clean samples
            perturbed_logits: Logits from perturbed samples
            
        Returns:
            KL divergence between distributions
        """
        clean_probs = F.log_softmax(clean_logits, dim=-1)
        perturbed_probs = F.softmax(perturbed_logits, dim=-1)
        
        loss = self.kl_loss(clean_probs, perturbed_probs)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss for end-to-end training.
    
    Combines task loss, distillation loss, and robustness loss.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        task_weight: float = 1.0,
        kd_weight: float = 0.5,
        robustness_weight: float = 0.3
    ):
        """Initialize combined loss.
        
        Args:
            temperature: Temperature for distillation
            task_weight: Weight for task loss
            kd_weight: Weight for distillation loss
            robustness_weight: Weight for robustness loss
        """
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
        """Compute combined loss.
        
        Args:
            student_logits: Student predictions
            teacher_logits: Teacher predictions
            targets: Ground truth labels
            clean_logits: Predictions on clean samples (for robustness)
            perturbed_logits: Predictions on perturbed samples (for robustness)
            
        Returns:
            Combined loss value
        """
        # Distillation loss
        kd_loss = self.distillation_loss(student_logits, teacher_logits, targets)
        
        # Robustness loss (optional)
        robustness_loss = 0.0
        if clean_logits is not None and perturbed_logits is not None:
            robustness_loss = self.robustness_loss(clean_logits, perturbed_logits)
        
        # Combined loss
        combined_loss = (
            self.task_weight * kd_loss +
            self.kd_weight * kd_loss +
            self.robustness_weight * robustness_loss
        )
        
        return combined_loss
