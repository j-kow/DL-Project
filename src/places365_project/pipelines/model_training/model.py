import torch
from torch import nn
from pytorch_lightning import LightningModule
import torchvision as tv
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class PlacesModel(LightningModule):
    def __init__(self, lr, patience, frequency, no_classes=365):
        super().__init__()
        self.save_hyperparameters()
        self.model = tv.models.mobilenet_v3_small(pretrained=True)

        # lock pretrained params
        for param in self.model.parameters():
            param.requires_grad = False

        # new classifier, should be unlocked by default
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=no_classes, bias=True),
        )

        with torch.no_grad():
            for name, p in self.model.classifier.named_parameters():
                if "weight" in name:
                    p.normal_(0, 0.1)
                elif "bias" in name:
                    p.normal_(0, 0.1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        if stage:
            self.log(stage+"_loss", loss, prog_bar=True)
            self.log(stage+"_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=self.hparams.patience),
                "monitor": "val_loss",
                "frequency": self.hparams.frequency
            },
        }
