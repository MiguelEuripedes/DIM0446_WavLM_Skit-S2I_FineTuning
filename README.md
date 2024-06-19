# Audio Intent Classification with WavLM

## Description
This project uses the WavLM model from Hugging Face for audio sequence classification. The goal is to automate the categorization of audio based on characteristics extracted by the model. For this, Fine Tuning of the [WavLM Model](https://huggingface.co/docs/transformers/model_doc/wavlm) was performed on the [Skit-S2I](https://github.com/skit-ai/speech-to-intent-dataset) database.

## Code Structure
The `trainer_wavML.py` script is responsible for the training process. It includes:

### Main Components
The project is based on PyTorch and PyTorch Lightning and uses the `WavLM` model for audio intent classification. The main libraries used include `torch`, `pandas`, `transformers`, `torchaudio`, and `pytorch_lightning`.

### Classes and Functions
- **`WavLMIntentClassifier`:** Defines the classification model based on WavLM, with methods to load the model and process input data.
- **`AudioDataset`:** A custom dataset for loading and preprocessing audio data. It uses resampling and tokenization to prepare the data for the model.
- **`LightningWavLMModel`:** A Lightning module that encapsulates the classification model, including definitions for training and validation steps.

### Data
- Data is loaded using the `AudioDataset` class, which reads audios and their respective classifications from a CSV.
- The data is split into training and validation sets.
- DataLoaders are set up to manage data loading during training and validation.

### Training and Validation
- Uses `ModelCheckpoint` to save the best models during training.
- PyTorch Lightning `Trainer` manages the training and validation loop, using GPUs when available. The model was trained for **5 epochs** to ensure good generalization without overfitting.
- Metrics such as loss and accuracy are calculated and recorded during training and validation.

### Evaluation
- After training, the best model is loaded and evaluated on the test set.
- Performance metrics such as accuracy and F1-Score are calculated to assess the effectiveness of the model.

### Utilities
- The code includes a standardization function (`collate_fn`) to ensure that data batches are correctly processed by the model.
- The script includes setting a seed (SEED) to ensure reproducibility.

## Training Results
The `slurm-504844.out` file provides logs from the training process.

```
Accuracy Score = 0.9583532790809
F1-Score = 0.9608108878176348
```

## References

The dataset was collected from the repository created by Professor Ranniery, for the university course about Deep Learning.

- [Professor Ranniery Maia's Repository](https://github.com/rdsmaia/speech-to-intent)
