from dataclasses import dataclass
import torchvision.transforms as T


@dataclass
class DatasetMnistConfig:
    train_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
