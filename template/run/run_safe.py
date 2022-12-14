from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

from template.evaluate.evaluator import Evaluator
from template.model.multi_modal.safe import SAFE
from template.data.dataset.safe_dataset import SAFEDataset
from template.train.eann_trainer import EANNTrainer


def run_safe(root: str):
    dataset = SAFEDataset(root_dir=Path(root))

    model = SAFE()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr=0.00025)
    evaluator = Evaluator(["accuracy", "precision", "recall", "f1"])

    trainer = EANNTrainer(model, evaluator, optimizer)
    trainer.fit(dataset,
                batch_size=20,
                epochs=100,
                validate_size=0.2,
                saved=True)


if __name__ == '__main__':
    root = ""
    run_safe(root, False)