import datetime
import os
from time import time
from typing import Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from template.evaluate.evaluator import Evaluator
from template.model.model import AbstractModel


class AbstractTrainer:
    def __init__(self, model: AbstractModel, evaluator: Evaluator,
                 optimizer: Optimizer, loss_func: Optional[Callable] = None):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, data: torch.utils.data.Dataset, batch_size: int):
        """evaluate after training"""
        raise NotImplementedError

    def fit(self, train_data: torch.utils.data.Dataset, batch_size: int,
            epochs: int, validate_data=None,
            validate_size=None, saved=False, save_path=None):
        raise NotImplementedError


class BaseTrainer:
    def __init__(self, model: AbstractModel, evaluator: Evaluator,
                 optimizer: Optimizer, loss_func: Optional[Callable] = None):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.evaluator = evaluator

    @torch.no_grad()
    def evaluate(self, data: torch.utils.data.Dataset, batch_size: int):
        self.model.eval()
        dataloader = DataLoader(data, batch_size, shuffle=False)
        outputs = []
        labels = []
        for batch_data in dataloader:
            outputs.append(self.model.predict(batch_data[:-1]))
            labels.append(batch_data[-1])
        return self.evaluator.evaluate(torch.concat(outputs), torch.concat(labels))

    def _train_epoch(self, data, batch_size: int, epoch: int) -> torch.float:
        """training for one epoch"""
        self.model.train()
        dataloader = DataLoader(data, batch_size, shuffle=True)
        # todo tqdm与print冲突
        # data_iter = tenumerate(
        #     dataloader,
        #     total=len(dataloader),
        #     ncols=100,
        #     desc=f'Training',
        #     leave=False
        # )
        msg = None
        for batch_id, batch_data in enumerate(dataloader):
            # using trainer.loss_func first or model.calculate_loss
            if self.loss_func is None:
                loss, msg = self.model.calculate_loss(batch_data)
            else:
                loss = self.loss_func(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if msg is not None:
            print(msg)
        return loss

    def _validate_epoch(self, data, batch_size: int):
        """validation after training for one epoch"""
        # todo 计算best score，作为最佳结果保存
        return self.evaluate(data, batch_size)

    def fit(self, train_data: torch.utils.data.Dataset, batch_size: int,
            epochs: int, validate_data=None,
            validate_size: Optional[float] = None, saved=False, save_path: Optional[str] = None):
        """training"""
        # whether split validation set
        validation = True
        if validate_data is None and validate_size is None:
            validation = False
        elif validate_data is None:
            validate_size = int(validate_size * len(train_data))
            train_size = len(train_data) - validate_size
            train_data, validate_data = torch.utils.data.random_split(
                train_data, [train_size, validate_size])

        # tqdm.write('----start training-----')
        print(f'training data size={len(train_data)}')
        if validation:
            print(f'validation data size={len(validate_data)}')

        # training for epochs
        print('----start training-----')
        for epoch in range(epochs):
            print(f'\n--epoch=[{epoch + 1}/{epochs}]--')
            training_start_time = time()
            training_loss = self._train_epoch(train_data, batch_size, epoch)
            training_end_time = time()
            print(f'time={training_end_time - training_start_time}s, '
                  f'train loss={training_loss}')
            if validation:
                validate_result = self._validate_epoch(validate_data,
                                                       batch_size)
                print(f'      validation result: {validate_result}')
            # tqdm.write(f'epoch={epoch}, '
            #            f'time={training_end_time - training_start_time}s, '
            #            f'train loss={training_loss}')
            # tqdm.write(f'validation result: {validate_result}')

        # save the model
        if saved:
            if save_path is None:
                save_dir = os.path.join(os.getcwd(), "save")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                file_name = f"{self.model.__class__.__name__}-{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.pth"
                save_path = os.path.join(save_dir, file_name)
            torch.save(self.model.state_dict(), save_path)
            print(f'model is saved as {save_path}')
