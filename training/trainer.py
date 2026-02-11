import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from adversarial.perturbations import AdversarialAttackGenerator
from config.config import Config
from distillation.hkd import HierarchicalKnowledgeDistillation


class Trainer:
    def __init__(
        self,
        config: Config,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        self.kd = HierarchicalKnowledgeDistillation(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=config.model.temperature,
            feature_alpha=float(getattr(config.training, "kd_feature_weight", 0.3)),
            logit_alpha=float(getattr(config.training, "kd_logit_weight", config.training.kd_loss_weight)),
            task_alpha=float(getattr(config.training, "kd_task_weight", config.training.vulnerability_loss_weight)),
            consistency_alpha=float(getattr(config.training, "consistency_loss_weight", config.training.robustness_loss_weight)),
            device=device,
        )

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.attack_generator = AdversarialAttackGenerator(
            perturbation_types=getattr(config.adversarial, "attack_types", None),
            seed=config.seed,
        )
        self.perturbation_budget = float(config.adversarial.perturbation_budget)
        self.perturbation_steps = max(1, int(getattr(config.adversarial, "attack_steps", 1)))
        self.use_robust_kd = bool(getattr(config.training, "use_robust_kd", True))
        self.clean_warmup_epochs = max(0, int(getattr(config.training, "clean_warmup_epochs", 0)))

        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.train_history = {"loss": [], "accuracy": []}
        self.val_history = {"loss": [], "accuracy": []}

        self.logger = logging.getLogger(__name__)

    def _create_optimizer(self) -> optim.Optimizer:
        params = self.kd.student.parameters()
        optimizer_name = self.config.training.optimizer.lower()
        if optimizer_name == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        if optimizer_name == "adam":
            return optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        if optimizer_name == "sgd":
            return optim.SGD(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

    def _create_scheduler(self) -> Optional[object]:
        scheduler_name = self.config.training.scheduler.lower()
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
            )
        if scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )
        if scheduler_name == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.training.num_epochs,
            )
        return None

    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        self.current_epoch += 1
        train_metrics = self._train_step(train_loader)
        val_metrics = self._val_step(val_loader) if val_loader is not None else {}

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_metrics = {**train_metrics, **val_metrics}
        self._log_epoch(epoch_metrics)
        self._check_stopping(val_metrics)
        return epoch_metrics

    def _train_step(self, train_loader: DataLoader) -> Dict[str, float]:
        self.kd.student.train()
        log_interval = max(1, int(getattr(self.config, "log_interval", 10)))
        use_robust = self.use_robust_kd and self.current_epoch > self.clean_warmup_epochs

        total_loss = 0.0
        total_correct = 0
        total_perturbed_correct = 0
        total_samples = 0
        num_batches = 0
        component_sums: Dict[str, float] = {}

        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)

            if use_robust:
                perturbed_batch = self._generate_perturbed_batch(batch)
                loss, student_out, loss_components = self.kd.forward_robust(
                    clean_graph_x=batch["graph_x"],
                    clean_edge_index=batch["graph_edge_index"],
                    targets=batch["label"],
                    clean_edge_attr=batch.get("graph_edge_attr"),
                    clean_batch=batch.get("graph_batch"),
                    perturbed_graph_x=perturbed_batch["graph_x"],
                    perturbed_edge_index=perturbed_batch["graph_edge_index"],
                    perturbed_edge_attr=perturbed_batch.get("graph_edge_attr"),
                    perturbed_batch=perturbed_batch.get("graph_batch"),
                    return_features=True,
                )
                logits = student_out["clean_logits"]
                perturbed_logits = student_out["perturbed_logits"]
            else:
                loss, student_out, loss_components = self.kd.forward(
                    graph_x=batch["graph_x"],
                    edge_index=batch["graph_edge_index"],
                    targets=batch["label"],
                    edge_attr=batch.get("graph_edge_attr"),
                    batch=batch.get("graph_batch"),
                    return_features=True,
                )
                logits = student_out["logits"]
                perturbed_logits = None

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.kd.student.parameters(),
                    self.config.training.gradient_clip,
                )
            self.optimizer.step()

            targets = batch["label"]
            preds = logits.argmax(dim=-1)

            total_loss += float(loss.item())
            total_correct += int((preds == targets).sum().item())
            total_samples += int(targets.size(0))
            num_batches += 1
            self.current_step += 1

            if perturbed_logits is not None:
                perturbed_preds = perturbed_logits.argmax(dim=-1)
                total_perturbed_correct += int((perturbed_preds == targets).sum().item())

            for key, value in loss_components.items():
                component_sums[key] = component_sums.get(key, 0.0) + self._to_float(value)

            if batch_idx % log_interval == 0:
                clean_acc = (total_correct / total_samples) if total_samples > 0 else 0.0
                if use_robust and total_samples > 0:
                    robust_acc = total_perturbed_correct / total_samples
                    self.logger.info(
                        f"Epoch {self.current_epoch} Batch {batch_idx}/{len(train_loader)} "
                        f"Loss: {loss.item():.4f} CleanAcc: {clean_acc:.4f} RobustAcc: {robust_acc:.4f}"
                    )
                else:
                    self.logger.info(
                        f"Epoch {self.current_epoch} Batch {batch_idx}/{len(train_loader)} "
                        f"Loss: {loss.item():.4f} Acc: {clean_acc:.4f}"
                    )

        if num_batches == 0 or total_samples == 0:
            return {
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "train_robust_accuracy": 0.0,
            }

        metrics: Dict[str, float] = {
            "train_loss": total_loss / num_batches,
            "train_accuracy": total_correct / total_samples,
        }
        if use_robust:
            metrics["train_robust_accuracy"] = total_perturbed_correct / total_samples

        for key, value in component_sums.items():
            metrics[f"train_{key}_loss"] = value / num_batches
        return metrics

    def _val_step(self, val_loader: DataLoader) -> Dict[str, float]:
        self.kd.student.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                loss, student_out, _ = self.kd.forward(
                    graph_x=batch["graph_x"],
                    edge_index=batch["graph_edge_index"],
                    targets=batch["label"],
                    edge_attr=batch.get("graph_edge_attr"),
                    batch=batch.get("graph_batch"),
                    return_features=False,
                )

                logits = student_out["logits"]
                preds = logits.argmax(dim=-1)
                targets = batch["label"]

                total_loss += float(loss.item())
                total_correct += int((preds == targets).sum().item())
                total_samples += int(targets.size(0))
                num_batches += 1

        if num_batches == 0 or total_samples == 0:
            return {
                "val_loss": 0.0,
                "val_accuracy": 0.0,
            }

        return {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_correct / total_samples,
        }

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _generate_perturbed_batch(self, batch: Dict) -> Dict:
        graph_batch = batch.get("graph_batch")
        if graph_batch is None or graph_batch.numel() == 0:
            x_pert, edge_index_pert, edge_attr_pert, info = self.attack_generator.generate_attack(
                batch["graph_x"],
                batch["graph_edge_index"],
                batch.get("graph_edge_attr"),
                perturbation_budget=self.perturbation_budget,
                num_steps=self.perturbation_steps,
            )
            return {
                "graph_x": x_pert,
                "graph_edge_index": edge_index_pert,
                "graph_edge_attr": edge_attr_pert,
                "graph_batch": graph_batch,
                "attack_info": info,
            }

        x = batch["graph_x"]
        edge_index = batch["graph_edge_index"]
        edge_attr = batch.get("graph_edge_attr")

        src = edge_index[0]
        dst = edge_index[1]
        num_graphs = int(graph_batch.max().item()) + 1

        x_parts = []
        edge_index_parts = []
        edge_attr_parts = []
        batch_parts = []
        edge_attr_missing = False
        node_offset = 0

        for graph_id in range(num_graphs):
            node_ids = torch.nonzero(graph_batch == graph_id, as_tuple=False).view(-1)
            if node_ids.numel() == 0:
                continue

            node_features = x.index_select(0, node_ids)
            global_to_local = torch.full(
                (x.size(0),),
                -1,
                dtype=torch.long,
                device=x.device,
            )
            global_to_local[node_ids] = torch.arange(
                node_ids.numel(),
                device=x.device,
                dtype=torch.long,
            )

            edge_mask = (graph_batch[src] == graph_id) & (graph_batch[dst] == graph_id)
            local_src = global_to_local[src[edge_mask]]
            local_dst = global_to_local[dst[edge_mask]]
            local_edge_index = torch.stack([local_src, local_dst], dim=0)

            local_edge_attr = None
            if edge_attr is not None and edge_attr.size(0) == edge_index.size(1):
                local_edge_attr = edge_attr[edge_mask]

            pert_x, pert_edge_index, pert_edge_attr, _ = self.attack_generator.generate_attack(
                node_features,
                local_edge_index,
                local_edge_attr,
                perturbation_budget=self.perturbation_budget,
                num_steps=self.perturbation_steps,
            )

            x_parts.append(pert_x)
            batch_parts.append(
                torch.full(
                    (pert_x.size(0),),
                    graph_id,
                    dtype=graph_batch.dtype,
                    device=graph_batch.device,
                )
            )
            edge_index_parts.append(pert_edge_index + node_offset)
            node_offset += pert_x.size(0)

            if pert_edge_attr is not None:
                edge_attr_parts.append(pert_edge_attr)
            else:
                edge_attr_missing = True

        if not x_parts:
            return {
                "graph_x": x.clone(),
                "graph_edge_index": edge_index.clone(),
                "graph_edge_attr": edge_attr.clone() if edge_attr is not None else None,
                "graph_batch": graph_batch.clone(),
            }

        pert_x = torch.cat(x_parts, dim=0)
        pert_edge_index = (
            torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        )
        pert_graph_batch = torch.cat(batch_parts, dim=0)

        if edge_attr_parts and not edge_attr_missing:
            pert_edge_attr = torch.cat(edge_attr_parts, dim=0)
        elif edge_attr is not None:
            attr_dim = edge_attr.size(1) if edge_attr.dim() > 1 else 1
            pert_edge_attr = torch.zeros(
                (pert_edge_index.size(1), attr_dim),
                dtype=edge_attr.dtype,
                device=edge_attr.device,
            )
        else:
            pert_edge_attr = None

        return {
            "graph_x": pert_x,
            "graph_edge_index": pert_edge_index,
            "graph_edge_attr": pert_edge_attr,
            "graph_batch": pert_graph_batch,
        }

    def _log_epoch(self, metrics: Dict[str, float]) -> None:
        log_msg = f"Epoch {self.current_epoch}: "
        for key, value in metrics.items():
            log_msg += f"{key}={value:.4f} "

            if "train" in key:
                metric_name = key.replace("train_", "")
                if metric_name not in self.train_history:
                    self.train_history[metric_name] = []
                self.train_history[metric_name].append(value)

            if "val" in key:
                metric_name = key.replace("val_", "")
                if metric_name not in self.val_history:
                    self.val_history[metric_name] = []
                self.val_history[metric_name].append(value)

        self.logger.info(log_msg)

    def _check_stopping(self, val_metrics: Dict[str, float]) -> None:
        if "val_loss" not in val_metrics:
            return

        val_loss = val_metrics["val_loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def should_stop(self) -> bool:
        if self.patience_counter >= self.config.training.patience:
            self.logger.info(
                f"Early stopping triggered after {self.config.training.patience} epochs"
            )
            return True
        if self.current_epoch >= self.config.training.num_epochs:
            return True
        return False

    def save_checkpoint(self, path: Path) -> None:
        checkpoint = {
            "epoch": self.current_epoch,
            "student_state_dict": self.kd.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.kd.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.train_history = checkpoint["train_history"]
        self.val_history = checkpoint["val_history"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.logger.info(f"Checkpoint loaded from {path}")

    @staticmethod
    def _to_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)
