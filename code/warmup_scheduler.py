import torch
import torch.nn as nn
import torch.optim as optim


class WarmupReduceLROnPlateau:
    def __init__(self, optimizer, scheduler, warmup_steps, initial_lr, target_lr):
        """
        Combinaison de warm-up et ReduceLROnPlateau.

        Args:
            optimizer: Optimizer de PyTorch.
            scheduler: Instance de torch.optim.lr_scheduler.ReduceLROnPlateau.
            warmup_steps: Nombre de steps pour le warm-up.
            initial_lr: Taux d'apprentissage initial.
            target_lr: Taux d'apprentissage cible après le warm-up.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.step_num = 0  # Compteur pour les steps

        # Met le LR initial
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

    def step(self, metrics=None):
        """
        Met à jour le taux d'apprentissage.

        Args:
            metrics: Critère pour ReduceLROnPlateau après le warm-up.
        """
        if self.step_num < self.warmup_steps:
            # Phase de warm-up
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.step_num / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.step_num += 1
        else:
            # Phase de ReduceLROnPlateau
            self.scheduler.step(metrics)

    def get_last_lr(self):
        """
        Retourne le dernier LR.
        """
        if self.step_num < self.warmup_steps:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.step_num / self.warmup_steps)
            return [lr]
        else:
            return self.scheduler._last_lr
