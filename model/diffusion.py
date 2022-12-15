from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST

from utils.beta_schedule import _extract_into_tensor, get_named_beta_schedule
from utils.utils import InfiniteDataLoader, show_first_batch, show_noising
from configs.dataset_mnist_config import DatasetMnistConfig
from configs.train_mnist_config import TrainMnistConfig


class Diffusion:
    def __init__(
        self,
        *,
        betas: np.array,
        loss_type: str = "mse"
    ):
        """
        Class that simulates Diffusion process. Does not store model or optimizer.
        """

        if loss_type == 'mse':
            self.loss = F.mse_loss
        else:
            raise('unknown loss type')

        betas = torch.from_numpy(betas).double()
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]), ], dim=0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)


        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # var from (3)
        self.posterior_variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * betas

        # log calculation clipped because posterior variance is 0.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]], dim=0)
        )
        # coef of xt from (2)
        self.posterior_mean_coef1 = alphas.sqrt() * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        # coef of x0 from (2)
        self.posterior_mean_coef2 = (self.alphas_cumprod_prev.sqrt() * betas) / (1 - self.alphas_cumprod)

    def q_mean_variance(self, x0, t):
        """
        Get mean and variance of distribution q(x_t | x_0). Use equation (1).
        """
        shp = x0.shape
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, shp) * x0
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, shp)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, shp)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute mean and variance of diffusion posterior q(x_{t-1} | x_t, x_0).
        Use equation (2) and (3).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (_extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t + 
                          _extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape) * x_start)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse data for a given number of diffusion steps.
        Sample from q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
                _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_mean_variance(self, model_output, x, t):
        """
        Apply model to get p(x_{t-1} | x_t). Use Equation (2) and plug in \hat{x}_0;
        """
        model_variance = torch.cat([self.posterior_variance[1:2], self.betas[1:]], dim=0)
        model_log_variance = torch.log(model_variance)
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        model_mean = (_extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * x + 
                          _extract_into_tensor(self.posterior_mean_coef2, t, pred_xstart.shape) * pred_xstart)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Get \hat{x0} from epsilon_{theta}. Use equation (4) to derive it.
        """
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def p_sample(self, model_output, x, t):
        """
        Sample from p(x_{t-1} | x_t).
        """
        # get mean, variance of p(xt-1|xt)
        out = self.p_mean_variance(model_output, x, t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        b, *_, device = *x.shape, x.device
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample}
    
    def p_sample_loop(self, model, shape, y_dist):
        """
        Samples a batch=shape[0] using diffusion model.
        """
        x = torch.randn(*shape, device=model.device)
        indices = list(range(self.num_timesteps))[::-1]

        y = torch.multinomial(
            y_dist,
            num_samples=shape[0],
            replacement=True
        ).to(x.device)

        for i in tqdm(indices):
            t = torch.tensor([i] * shape[0], device=x.device)
            with torch.no_grad():
                model_output = model(x, t, y)
                out = self.p_sample(
                    model_output,
                    x,
                    t
                )
                x = out["sample"]
        return x, y
    
    def train_loss(self, model, x0, y):
        """
        Calculates loss L^{simple}_t for the given model, x0.
        """
        t = torch.randint(0, self.num_timesteps, size=(x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        model_output = model(x_t, t, y)
        loss = self.loss(noise, model_output)
        return loss


def test_1st_batch():
    dataset = MNIST("../datasets", download=True, train=True, transform=DatasetMnistConfig.train_transforms)
    loader = InfiniteDataLoader(dataset, batch_size=512, shuffle=True,
                                num_workers=TrainMnistConfig.n_workers, pin_memory=True)
    show_first_batch(loader)


def test_noise():
    dataset = MNIST("../datasets", download=True, train=True, transform=DatasetMnistConfig.train_transforms)
    T = 1000
    for sch in ["linear", "quad", "sigmoid"]:
        diffusion_temp = Diffusion(
            betas=get_named_beta_schedule(sch, T),
            loss_type="mse"
        )
        show_noising(diffusion_temp, dataset[0][0])


if __name__ == '__main__':
    test_1st_batch()
    # test_noise()
