# DDPM

My implementation of "Denoising Diffusion Probabilistic Models" paper

[Paper DPPM](https://arxiv.org/pdf/2006.11239.pdf)

Code based on HSE DL [homework](https://github.com/mryab/dl-hse-ami/blob/main/week10_probmodels/homework.ipynb)

# Reproduce code

1. ```python
   git clone https://github.com/messlav/ddpm.git
   cd ddpm
   pip install -r requirements.txt
   ```
2.  ```python
    python3 train.py
    ```

# Report 

I'm using [this U-Net](https://github.com/mryab/dl-hse-ami/blob/main/week10_probmodels/utils.py) – 
implementation with LayerNorm, sinusoidal position embeddingб and self-attention

Batch size = 512, AdamW optimizer without scheduler, sigmoid beta schedule, weight decay = 1e-4.

I conducted three experiments. The first experiment used a learning rate = 1e-3 and 10.000 steps:


Second one with learning rate = 1e-5 and 20.000 steps:


That was bad idea, therefore, I decided to decrease batch size to 128 and
set learning rate to 1e-3 with 10.000 steps:

MNIST is not a popular dataset for such powerful models like DPPM, herefore, there are not many examples
of generated images on the internet. Furthermore, the generated images do not look very good, possibly because 
of the choice of hyperparameters