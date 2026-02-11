import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .losses import HierarchicalDistillationLoss


class HierarchicalKnowledgeDistillation:
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        feature_alpha: float = 0.3,
        logit_alpha: float = 0.5,
        task_alpha: float = 1.0,
        consistency_alpha: float = 0.0,
        device: str = "cuda",
    ):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device

        self.criterion = HierarchicalDistillationLoss(
            temperature=temperature,
            feature_alpha=feature_alpha,
            logit_alpha=logit_alpha,
            task_alpha=task_alpha,
            consistency_alpha=consistency_alpha,
        )
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _unfreeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad = True

    def forward(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        targets: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        with torch.no_grad():
            if return_features:
                teacher_logits, _ = self.teacher.forward(
                    graph_x,
                    edge_index,
                    edge_attr,
                    batch,
                    return_features=True,
                )
                teacher_layer_features = self.teacher.get_intermediate_representations(
                    graph_x,
                    edge_index,
                    edge_attr,
                    batch,
                )
            else:
                teacher_logits, _ = self.teacher.forward(
                    graph_x,
                    edge_index,
                    edge_attr,
                    batch,
                    return_features=False,
                )
                teacher_layer_features = None

        if return_features:
            student_logits, student_features = self.student.forward(
                graph_x,
                edge_index,
                edge_attr,
                batch,
                return_features=True,
            )
            student_layer_features = self.student.get_intermediate_representations(
                graph_x,
                edge_index,
                edge_attr,
                batch,
            )
        else:
            student_logits, _ = self.student.forward(
                graph_x,
                edge_index,
                edge_attr,
                batch,
                return_features=False,
            )
            student_features = None
            student_layer_features = None

        loss, loss_components = self.criterion(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            student_features=student_layer_features,
            teacher_features=teacher_layer_features,
        )

        student_outputs = {
            "logits": student_logits,
            "features": student_features if return_features else None,
            "teacher_logits": teacher_logits,
        }
        return loss, student_outputs, loss_components

    def forward_robust(
        self,
        clean_graph_x: torch.Tensor,
        clean_edge_index: torch.Tensor,
        targets: torch.Tensor,
        clean_edge_attr: Optional[torch.Tensor] = None,
        clean_batch: Optional[torch.Tensor] = None,
        perturbed_graph_x: Optional[torch.Tensor] = None,
        perturbed_edge_index: Optional[torch.Tensor] = None,
        perturbed_edge_attr: Optional[torch.Tensor] = None,
        perturbed_batch: Optional[torch.Tensor] = None,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        if perturbed_graph_x is None or perturbed_edge_index is None:
            raise ValueError("perturbed_graph_x and perturbed_edge_index are required for robust forward")

        with torch.no_grad():
            if return_features:
                teacher_logits, _ = self.teacher.forward(
                    clean_graph_x,
                    clean_edge_index,
                    clean_edge_attr,
                    clean_batch,
                    return_features=True,
                )
                teacher_layer_features = self.teacher.get_intermediate_representations(
                    clean_graph_x,
                    clean_edge_index,
                    clean_edge_attr,
                    clean_batch,
                )
            else:
                teacher_logits, _ = self.teacher.forward(
                    clean_graph_x,
                    clean_edge_index,
                    clean_edge_attr,
                    clean_batch,
                    return_features=False,
                )
                teacher_layer_features = None

        if return_features:
            student_clean_logits, student_clean_features = self.student.forward(
                clean_graph_x,
                clean_edge_index,
                clean_edge_attr,
                clean_batch,
                return_features=True,
            )
            student_perturbed_logits, student_perturbed_features = self.student.forward(
                perturbed_graph_x,
                perturbed_edge_index,
                perturbed_edge_attr,
                perturbed_batch if perturbed_batch is not None else clean_batch,
                return_features=True,
            )
            student_clean_layers = self.student.get_intermediate_representations(
                clean_graph_x,
                clean_edge_index,
                clean_edge_attr,
                clean_batch,
            )
            student_perturbed_layers = self.student.get_intermediate_representations(
                perturbed_graph_x,
                perturbed_edge_index,
                perturbed_edge_attr,
                perturbed_batch if perturbed_batch is not None else clean_batch,
            )
        else:
            student_clean_logits, _ = self.student.forward(
                clean_graph_x,
                clean_edge_index,
                clean_edge_attr,
                clean_batch,
                return_features=False,
            )
            student_perturbed_logits, _ = self.student.forward(
                perturbed_graph_x,
                perturbed_edge_index,
                perturbed_edge_attr,
                perturbed_batch if perturbed_batch is not None else clean_batch,
                return_features=False,
            )
            student_clean_features = None
            student_perturbed_features = None
            student_clean_layers = None
            student_perturbed_layers = None

        loss, loss_components = self.criterion.forward_robust(
            student_clean_logits=student_clean_logits,
            student_perturbed_logits=student_perturbed_logits,
            teacher_clean_logits=teacher_logits,
            targets=targets,
            student_clean_features=student_clean_layers,
            student_perturbed_features=student_perturbed_layers,
            teacher_clean_features=teacher_layer_features,
        )

        student_outputs = {
            "clean_logits": student_clean_logits,
            "perturbed_logits": student_perturbed_logits,
            "clean_features": student_clean_features if return_features else None,
            "perturbed_features": student_perturbed_features if return_features else None,
            "teacher_logits": teacher_logits,
        }
        return loss, student_outputs, loss_components

    def get_teacher_features(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        layer_indices: Optional[list] = None,
    ) -> Dict:
        with torch.no_grad():
            return self.teacher.get_intermediate_representations(
                graph_x,
                edge_index,
                edge_attr,
                batch,
                layer_indices,
            )

    def get_student_features(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        layer_indices: Optional[list] = None,
    ) -> Dict:
        return self.student.get_intermediate_representations(
            graph_x,
            edge_index,
            edge_attr,
            batch,
            layer_indices,
        )

    def analyze_distillation(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        targets: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict:
        with torch.no_grad():
            teacher_logits, _ = self.teacher(
                graph_x,
                edge_index,
                edge_attr,
                batch,
            )
            student_logits, _ = self.student(
                graph_x,
                edge_index,
                edge_attr,
                batch,
            )

            teacher_probs = torch.softmax(teacher_logits, dim=-1)
            student_probs = torch.softmax(student_logits, dim=-1)

            teacher_preds = teacher_logits.argmax(dim=-1)
            student_preds = student_logits.argmax(dim=-1)
            agreement = (teacher_preds == student_preds).float().mean()

            kl_div = torch.nn.functional.kl_div(
                torch.log_softmax(student_logits / 4.0, dim=-1),
                torch.softmax(teacher_logits / 4.0, dim=-1),
                reduction="batchmean",
            )

            return {
                "teacher_accuracy": (teacher_preds == targets).float().mean().item(),
                "student_accuracy": (student_preds == targets).float().mean().item(),
                "agreement": agreement.item(),
                "kl_divergence": kl_div.item(),
                "confidence_student": student_probs.max(dim=-1)[0].mean().item(),
                "confidence_teacher": teacher_probs.max(dim=-1)[0].mean().item(),
            }
