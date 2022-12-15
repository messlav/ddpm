import torch
from torchvision.datasets.mnist import MNIST
from argparse import ArgumentParser
import os

from model.diffusion import Diffusion
from model.simple_unet import MyUNet
from trainer.trainer import Trainer
from configs.train_mnist_config import TrainMnistConfig
from configs.dataset_mnist_config import DatasetMnistConfig
from utils.beta_schedule import get_named_beta_schedule
from utils.utils import InfiniteDataLoader, sample_synthetic, show_images
import matplotlib.pyplot as plt


def main(name):
    scheduler = 'sigmoid'
    train_config = TrainMnistConfig
    datset_config = DatasetMnistConfig

    os.makedirs('losses', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('generated_examples', exist_ok=True)

    dataset = MNIST("./datasets", download=True, train=True, transform=datset_config.train_transforms)
    loader = InfiniteDataLoader(dataset, batch_size=512, shuffle=True,
                                num_workers=train_config.n_workers, pin_memory=True)

    diffusion_mnist = Diffusion(
        betas=get_named_beta_schedule(scheduler, 1000),
        loss_type="mse"
    )

    model_mnist = MyUNet().to(train_config.device)
    model_mnist.device = train_config.device

    trainer_mnist = Trainer(
        diffusion_mnist,
        model_mnist,
        train_iter=loader,
        lr=1e-3,
        steps=100,
        weight_decay=1e-4,
        device=TrainMnistConfig.device
    )
    losses = trainer_mnist.run_loop()
    plt.plot(losses)
    plt.xlabel('steps', fontsize=15)
    plt.ylabel('L simple', fontsize=15)
    plt.grid()
    plt.savefig(f'losses/{name}.png')
    plt.show()
    torch.save({'model': model_mnist.state_dict(), 'optimizer': trainer_mnist.optimizer.state_dict()},
               f'weights/{name}.pth.tar')

    Xs, ys = sample_synthetic(
        diffusion_mnist,
        model_mnist,
        num_samples=16,
        batch=16,
        shape=(1, 28, 28),
        y_dist=[0.1 for _ in range(10)]
    )
    show_images(Xs, ys, f'generated_examples/{name}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--name", type=str, default='name_me_plz', help="name of run")
    args = parser.parse_args()
    main(args.name)
