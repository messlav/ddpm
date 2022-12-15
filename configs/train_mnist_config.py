from dataclasses import dataclass
import torch


@dataclass
class TrainMnistConfig:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_workers = 4
    batch_size = 1
