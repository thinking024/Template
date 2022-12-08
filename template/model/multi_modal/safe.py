import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# convolution layers
class ConvBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_len: int,
        num_filters: int,
        dropout_prob: float,
        final_len: int,
    ):
        super().__init__()

        # filter size: 3
        self.conv3 = nn.Conv2d(1, num_filters, kernel_size=(3, input_len))
        self.pool3 = nn.MaxPool2d(
            kernel_size=(input_size - 3 + 1, 1),
            stride=(1, 1),
        )

        # filter size: 4
        self.conv4 = nn.Conv2d(1, num_filters, kernel_size=(4, input_len))
        self.pool4 = nn.MaxPool2d(
            kernel_size=(input_size - 4 + 1, 1),
            stride=(1, 1),
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(num_filters * 2, final_len)

    def forward(self, x: torch.Tensor):
        # bs, head_size, input_len -> bs, 1, head_size, input_len
        x = x.unsqueeze(1)

        # filter size: 3
        x3 = self.conv3(x)
        x3 = self.pool3(x3)

        # filter size: 3
        x4 = self.conv4(x)
        x4 = self.pool4(x4)

        x = torch.cat([x3, x4], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, x.shape[-1])

        x = self.dropout(x)

        return self.fc(x)


class SAFE(pl.LightningModule):
    def __init__(
        self,
        head_size: int = 30,
        body_size: int = 100,
        image_size: int = 66,
        embedding_size: int = 300,
        input_len: int = 32,
        num_filters: int = 128,
        final_len: int = 200,
        learning_rate: float = 1e-3,
        max_epochs: int = 30,
        dropout_prob: float = 0.,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding_size = embedding_size
        self.input_len = input_len
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.dropout_prob = dropout_prob

        self.reduce = nn.Linear(embedding_size, input_len)
        nn.init.trunc_normal_(self.reduce.weight, std=0.1)
        nn.init.constant_(self.reduce.bias, 0.1)

        self.head_block = ConvBlock(
            input_size=head_size,
            input_len=input_len,
            num_filters=num_filters,
            dropout_prob=dropout_prob,
            final_len=final_len,
        )

        self.body_block = ConvBlock(
            input_size=body_size,
            input_len=input_len,
            num_filters=num_filters,
            dropout_prob=dropout_prob,
            final_len=final_len,
        )

        self.image_block = ConvBlock(
            input_size=image_size,
            input_len=input_len,
            num_filters=num_filters,
            dropout_prob=dropout_prob,
            final_len=final_len,
        )

        self.predictor = nn.Linear(final_len * 3, 2)
        nn.init.trunc_normal_(self.predictor.weight, std=0.1)
        nn.init.constant_(self.predictor.bias, 0.1)

    def forward(
        self,
        x_heads: torch.Tensor,
        x_bodies: torch.Tensor,
        x_images: torch.Tensor,
        targets: torch.Tensor,
    ):
        # 全连接层降维
        x_heads = self.reduce(x_heads)
        x_bodies = self.reduce(x_bodies)
        x_images = self.reduce(x_images)

        # 卷积层
        headline_vectors = self.head_block(x_heads)
        body_vectors = self.body_block(x_bodies)
        image_vectors = self.image_block(x_images)

        # cos similarity
        combine_images = torch.cat([image_vectors, image_vectors], dim=1)
        combine_texts = torch.cat([headline_vectors, body_vectors], dim=1)

        combine_images_norm = combine_images.norm(p=2, dim=1)
        combine_texts_norm = combine_texts.norm(p=2, dim=1)

        image_text = (combine_images * combine_texts).sum(1)

        cosine_similarity = (
            1 + (image_text / (combine_images_norm * combine_texts_norm + 1e-8))
        ) / 2
        distance = 1 - cosine_similarity
        cos_preds = torch.stack([distance, cosine_similarity], 1)

        # 全连接层预测
        vecs = torch.cat([headline_vectors, body_vectors, image_vectors], dim=1)
        logits = self.predictor(vecs)

        # loss
        loss1 = F.cross_entropy(logits, targets)
        loss2 = -(targets * cos_preds.log()).sum(1).mean()

        loss = loss1 * 0.6 + loss2 * 0.4

        preds = logits.argmax(1)
        labels = targets.argmax(1)
        acc = (preds == labels).sum().item() / logits.shape[0] * 100

        return loss, acc

    def training_step(self, batch, batch_idx: int):
        x_heads, x_bodies, x_images, targets = batch

        loss, acc = self.forward(x_heads, x_bodies, x_images, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train/acc", acc,
            on_step=True, on_epoch=True, prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        x_heads, x_bodies, x_images, targets = batch

        loss, acc = self.forward(x_heads, x_bodies, x_images, targets)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log(
            "val/acc", acc,
            on_step=True, on_epoch=True, prog_bar=True,
        )

    def test_step(self, batch, batch_idx: int):
        ...

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.max_epochs
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": lr_scheduler,
        }
