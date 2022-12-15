import torch
import torch.nn as nn
import numpy as np

from model.diffusion import Diffusion


class Trainer:
    def __init__(
        self,
        diffusion: Diffusion,
        model: nn.Module,
        train_iter, # iterable that yields (x, y)
        lr: float,
        weight_decay: float,
        steps: int,
        device: torch.device = torch.device('cpu')
    ):
        self.diffusion = diffusion

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.log_every = 100
        self.print_every = 500

    def _anneal_lr(self, step: int):
        """
        Performs annealing of lr.
        """

        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x: torch.FloatTensor, y: torch.LongTensor):
        """
        A single training step.
        Calculates loss for a single batch. 
        Then performs a single optimizer step and returns loss.
        """
        self.optimizer.zero_grad()

        loss = self.diffusion.train_loss(self.model, x, y)
        # print(loss)

        loss.backward()
        self.optimizer.step()

        return loss

    def run_loop(self):
        """
        Training loop.
        """
        step = 0
        curr_loss_gauss = 0.0

        curr_count = 0
        all_loses = []
        while step < self.steps:
            x, y = next(self.train_iter)
            x = x.to(self.device)
            y = y.to(self.device)
            batch_loss = self._run_step(x, y)

            self._anneal_lr(step)

            curr_count += len(x)
            loss = batch_loss.item()
            curr_loss_gauss += loss * len(x)
            all_loses += [loss]

            if (step + 1) % self.log_every == 0:
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} Loss: {gloss}')
                curr_count = 0
                curr_loss_gauss = 0.0

            step += 1

        return all_loses
