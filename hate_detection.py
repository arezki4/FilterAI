import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics import AUROC
from pytorch_lightning.callbacks import RichProgressBar
#from pytorch_lightning.metrics.functional.classification import auroc
from sklearn.model_selection import train_test_split


print("importation fini")
df = pd.read_csv("content/train-3.csv")


print(torch.cuda.is_available())
print(torch.version.cuda)

print('df chargé')
df.head()

df.shape

df.isnull().sum()

train_df, test_df = train_test_split(df, test_size=0.1)
train_df.shape, test_df.shape

CLASSES = df.columns.to_list()[2:]
CLASSES

df[CLASSES].sum().sort_values().plot(kind="barh")

df[CLASSES].sum(), df.shape

toxic_df = df[df[CLASSES].sum(axis=1) > 0]
toxic_df.shape

# commentaire jugé clean ( no hate)
clean_df = df[df[CLASSES].sum(axis=1) == 0]
clean_df.shape

#pour &quilibrer un peu les donnees
clean_df = clean_df.sample(16_000)
clean_df.shape

# prendre un echantillons "équilibré" entre clean et hate comms
train_df = pd.concat([toxic_df, clean_df])
train_df.shape

train_df[CLASSES].sum()
print(" echantillon fait ")


BERT_MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

"""###**Dataset Preparation**"""

class ToxicCommentsDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_len: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        single_row = self.data.iloc[index]

        comment = single_row.comment_text
        labels = single_row[CLASSES]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "comment_text": comment,
            "input_ids": encoding["input_ids"].flatten(), # [1,512] => [512]
            "attention_mask": encoding["attention_mask"].flatten(), # [1,512] => [512]
            "labels": torch.FloatTensor(labels)
        }

train_dataset = ToxicCommentsDataset(train_df, tokenizer)

sample_data = train_dataset[0]

print(sample_data["comment_text"])
print()
print(sample_data["input_ids"])
print()
print(sample_data["attention_mask"])
print()
print(sample_data["labels"])

class ToxicCommentDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_len=128):
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    def setup(self,stage=None):
        self.train_dataset = ToxicCommentsDataset(self.train_df, self.tokenizer, self.max_len)
        self.test_dataset = ToxicCommentsDataset(self.test_df, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
        )

EPOCHS = 20
BATCH_SIZE = 32

data_module = ToxicCommentDataModule(
    train_df,
    test_df,
    tokenizer,
    batch_size=BATCH_SIZE
)
data_module.setup()

"""###**MODEL BUILDING**"""

class ToxicCommentClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, steps_per_epoch=None, n_epochs=None):
        super().__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "predictions": outputs,
            "labels": labels
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

   

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]

model = ToxicCommentClassifier(
    n_classes=len(CLASSES),
    steps_per_epoch=len(train_df)//BATCH_SIZE,
    n_epochs=EPOCHS
)


print("Model et tokenizer initialisés")

# trainer = pl.Trainer(max_epochs=EPOCHS, accelerator='gpu', callbacks=[RichProgressBar()])

# trainer.fit(model, data_module)
# print("l'entrainement est fini")

# torch.save(model.state_dict(), 'modele2_pl.pt')
# print("le model est saved")

# trainer.test()

# trainer.save_checkpoint("last-checkpoint.ckpt")
# trainer.save("./")
# print("saved")
"""###**Predictions**"""

# trained_model = ToxicCommentClassifier.load_from_checkpoint("last-checkpoint.ckpt", n_classes=len(CLASSES))
# trained_model.freeze()
# #### for testinnng 
# test_example = "I dont like you, I hate your texts those are really bullshit!"

# encoding = tokenizer.encode_plus(
#    test_example,
#    add_special_tokens=True,
#    max_length=128,
#    return_token_type_ids=False,
#    padding="max_length",
#    truncation=True,
#    return_attention_mask=True,
#    return_tensors="pt"
# )

# model.eval()
# _, preds = model(encoding["input_ids"], encoding["attention_mask"])
# preds = preds.flatten().detach().numpy()

# predictions = []
# for idx, label in enumerate(CLASSES):
#    if preds[idx] > 0.5:
#        predictions.append((label, round(preds[idx]*100, 2)))

# predictions

# print(predictions)