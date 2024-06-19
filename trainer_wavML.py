import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, WavLMForSequenceClassification
import os
import pandas as pd
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

class WavLMIntentClassifier(nn.Module):
	def __init__(self, model_name="microsoft/wavlm-base", num_classes=14):
		super().__init__()
		self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
		self.model = WavLMForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

		# Freeze the feature extractor layers
		#for param in self.model.wavlm.feature_extractor.parameters():
		#	param.requires_grad = False

	def forward(self, input_values, **kwargs):
		return self.model(input_values, **kwargs)

def collate_fn(batch):
	inputs, labels = zip(*batch)
	inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
	labels = torch.tensor(labels)
	return inputs, labels

class AudioDataset(torch.utils.data.Dataset):
	def __init__(self, csv_path, wav_dir):
		self.df = pd.read_csv(csv_path)
		self.wav_dir = wav_dir
		self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
		self.resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		audio_path = os.path.join(self.wav_dir, row["audio_path"])
		intent_class = int(row["intent_class"])

		speech_array, sampling_rate = torchaudio.load(audio_path)
		speech_array = speech_array.squeeze().numpy()

		if sampling_rate != 16000:
			speech_array = self.resampler(torch.tensor(speech_array).unsqueeze(0)).squeeze(0).numpy()

		inputs = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
		inputs = inputs.squeeze(0)

		return inputs, intent_class

class LightningWavLMModel(pl.LightningModule):
	def __init__(self, model_name="microsoft/wavlm-base", num_classes=14):
		super().__init__()
		self.model = WavLMIntentClassifier(model_name, num_classes)

	def forward(self, x):
		return self.model(x)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-5)

	def training_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = F.cross_entropy(logits.logits, y)
		acc = (logits.logits.argmax(dim=1) == y).float().mean()

		self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = F.cross_entropy(logits.logits, y)
		acc = (logits.logits.argmax(dim=1) == y).float().mean()

		self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

		return loss


# Seed
SEED = 100
torch.manual_seed(SEED)

# Dataset
dataset = AudioDataset(
	csv_path="/home/mendamaral/corporas/speech-to-intent/train.csv",
	wav_dir="/home/mendamaral/corporas/speech-to-intent"
)

# Train-validation Split
train_len = int(len(dataset) * 0.90)
val_len = len(dataset) - train_len
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

# DataLoaders
trainloader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=2,
	shuffle=True,
	num_workers=2,
	collate_fn=collate_fn,
)

valloader = torch.utils.data.DataLoader(
	val_dataset,
	batch_size=2,
	num_workers=2,
	collate_fn=collate_fn,
)

# Model
model = LightningWavLMModel()
run_name = "wavLM-intent-classification"

# Model checkpoints
model_checkpoint_callback = ModelCheckpoint(
	dirpath='checkpoints',
	monitor='val/acc',
	mode='max',
	verbose=1,
	filename=run_name+"-epoch={epoch}.ckpt"
)

# Trainer
trainer = Trainer(
	fast_dev_run=False,
	devices=1,
	accelerator='gpu',
	max_epochs=5,
	callbacks=[model_checkpoint_callback],
)

trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

# Test Dataset
test_dataset = AudioDataset(
	csv_path="/home/mendamaral/corporas/speech-to-intent/train.csv",
	wav_dir="/home/mendamaral/corporas/speech-to-intent"
)

# Load best checkpoints
best_model_path = trainer.checkpoint_callback.best_model_path
model = LightningWavLMModel.load_from_checkpoint(best_model_path)
model.to('cuda')
model.eval()

trues, preds = [], []

for x, label in tqdm(test_dataset):
	x_tensor = x.to("cuda").unsqueeze(0)
	with torch.no_grad():
		y_hat = model(x_tensor)
	probs = F.softmax(y_hat.logits, dim=1).detach().cpu()
	pred = probs.argmax(dim=1).numpy()
	trues.append(label)
	preds.append(pred[0])

print(f"\n\nAccuracy Score = {accuracy_score(trues, preds)}")
print(f"F1-Score = {f1_score(trues, preds, average='weighted')}")

from huggingface_hub import Repository
model.model.model.push_to_hub(run_name)
