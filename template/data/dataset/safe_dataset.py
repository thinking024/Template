from pathlib import Path

import numpy as np
import torch.utils.data


class SAFEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Path,
    ):
        super().__init__()

        x_heads = np.load(root_dir / "case_headline.npy", allow_pickle=True)
        x_bodies = np.load(root_dir / "case_body.npy", allow_pickle=True)
        x_images = np.load(root_dir / "case_image.npy", allow_pickle=True)
        y = np.load(root_dir / "case_y_fn.npy")

        self.x_heads = x_heads.astype(np.float32)
        self.x_bodies = x_bodies.astype(np.float32)
        self.x_images = x_images.astype(np.float32)
        self.y = y.astype(np.float32)

        assert self.x_heads.shape[0] == self.x_bodies.shape[0] == self.x_images.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.x_heads.shape[0]

    def __getitem__(self, index: int):
        return self.x_heads[index], self.x_bodies[index], self.x_images[index], self.y[index]
