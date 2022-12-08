import pytorch_lightning as pl
# from data import NumpyDatasets 数据及格式暂未设置
from model.multi_modal.safe import SAFE


def main():
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=60,
    )
    trainer.fit(
        datamodule=NumpyDatasets( # 数据及格式暂未设置
            root_path=r"D:\NPU_project\SAFE_Modified",
            train_batch_size=64,
            val_batch_size=64,
        ),
        model=SAFE(
            learning_rate=5e-4,
            max_epochs=60,
        ),
    )


if __name__ == "__main__":
    main()
