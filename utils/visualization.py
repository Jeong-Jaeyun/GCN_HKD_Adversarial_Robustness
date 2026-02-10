
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def plot_robustness_curve(
    results: Dict,
    save_path: Optional[Path] = None,
    title: str = "Adversarial Robustness Curve"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    budgets = sorted(results['adversarial'].keys())


    ax = axes[0]
    attack_types = set()
    for budget_results in results['adversarial'].values():
        attack_types.update(budget_results.keys())

    for attack_type in attack_types:
        accs = []
        for budget in budgets:
            if attack_type in results['adversarial'][budget]:
                acc = results['adversarial'][budget][attack_type]['accuracy_perturbed']
                accs.append(acc)

        ax.plot(budgets, accs, marker='o', label=attack_type)


    clean_acc = results['clean']['accuracy']
    ax.axhline(y=clean_acc, color='green', linestyle='--', label='Clean Accuracy')

    ax.set_xlabel('Perturbation Budget')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Perturbation Budget')
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[1]
    for attack_type in attack_types:
        gaps = []
        for budget in budgets:
            if attack_type in results['adversarial'][budget]:
                gap = results['adversarial'][budget][attack_type]['robustness_gap']
                gaps.append(gap)

        ax.plot(budgets, gaps, marker='s', label=attack_type)

    ax.set_xlabel('Perturbation Budget')
    ax.set_ylabel('Robustness Gap')
    ax.set_title('Robustness Gap vs Perturbation Budget')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_training_history(
    train_history: Dict[str, List],
    val_history: Dict[str, List],
    save_path: Optional[Path] = None,
    title: str = "Training History"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    if 'loss' in train_history and 'loss' in val_history:
        ax = axes[0]
        epochs = range(1, len(train_history['loss']) + 1)
        ax.plot(epochs, train_history['loss'], 'b-', label='Train Loss')
        ax.plot(epochs, val_history['loss'], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)


    if 'accuracy' in train_history and 'accuracy' in val_history:
        ax = axes[1]
        epochs = range(1, len(train_history['accuracy']) + 1)
        ax.plot(epochs, train_history['accuracy'], 'b-', label='Train Accuracy')
        ax.plot(epochs, val_history['accuracy'], 'r-', label='Val Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['Safe', 'Vulnerable'],
    save_path: Optional[Path] = None,
    normalize: bool = True
) -> None:
    import matplotlib.pyplot as plt

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)


    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
