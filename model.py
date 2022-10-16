from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim.lr_scheduler import MultiplicativeLR
import pytorch_lightning as pl
import torch




class T5(pl.LightningModule):

    def __init__(
        self,
        learning_rate=2e-5,
        multiply_lr_step=0.7,
        warmup_steps=100.0,
        model_path="t5-small",
        model_save_directory="t5-small.pkl",
        max_source_length=512,
        max_target_length=128,
        model_load_directory=None,
        dev=None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_save_directory = model_save_directory
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.warmup_steps = warmup_steps
        self.multiply_lr_step = multiply_lr_step
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.dev = dev


    def forward(self, input_sequences, output_sequences, **kwargs):
        input_sequences = [sequence for sequence in input_sequences]
        input_tokens = self.tokenizer(
            input_sequences,
            padding=True,
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = input_tokens.input_ids
        attention_mask = input_tokens.attention_mask

        target_encoding = self.tokenizer(
            output_sequences,
            padding=True,
            max_length=self.max_target_length,
            truncation=True,
        )
        labels = target_encoding.input_ids
        labels = [
            [(label if label != self.tokenizer.pad_token_id else -100) for label in labels_example]
            for labels_example in labels
        ]
        labels = torch.tensor(labels)

        loss = self.model(
            input_ids=input_ids.to(self.dev),
            attention_mask=attention_mask.to(self.dev),
            labels=labels.to(self.dev),
        ).loss
        return loss

    def training_step(self, batch, batch_idx):
        input_sequences, output_sequences = batch["input"], batch["target"]
        loss = self(input_sequences, output_sequences)
        self.log("loss", loss, logger=True, batch_size=1)
        self.log("lr", self.scheduler._last_lr[-1], logger=True, batch_size=1)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        if self.trainer.global_step > 0:
            print("Saving model...")
            torch.save(self.model.state_dict(), self.model_save_directory)
            self.scheduler.step()

    def validation_step(self, batch, batch_idx):
        input_sequences, output_sequences = batch["input"], batch["target"]
        loss = self(input_sequences, output_sequences)
        self.log("validation_loss", loss, logger=True, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def lambd(epoch):
            return self.multiply_lr_step

        self.scheduler = MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [self.scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

        optimizer.step(closure=optimizer_closure)

    def save(self):
        torch.save(self.model.state_dict(), self.model_save_directory)

