# Rebuilding GPT-2 from Scratch

This repository contains an implementation of a small GPT-2-like model, built from scratch.

## Project Structure

- **gpt_architecture/** - Contains the core model architecture.
- **pretraining/** - Logic for pretraining the model.
- **fine-tuning/** - Methods for loading and finetuning the pretrained model on specific datasets.
- **train.py** - Script for pretraining a small GPT-2-like model on a simple text corpus (`the-verdict.txt`).
- **load_pretrain.py** - Loads the pretrained GPT-2 (124M) weights into the custom architecture for evaluation and experimentation.

## Getting Started

### Installation
Clone the repository and install the required dependencies:
```sh
 git clone https://github.com/kennethSty/Playground.git
 cd Playground
 pip install -r requirements.txt
```

### Training the Model
To pretrain the model from scratch using `train.py`, run:
```sh
python train.py
```

### Loading Pretrained GPT-2 Weights
To load the published GPT-2 (124M) model weights into your architecture, run:
```sh
python load_pretrain.py
```

### Finetunign for Classification (in progress)

## Credits
This project is heavily inspired by Sebastian Raschka’s book *Building LLMs from Scratch*. If you're interested in understanding the inner workings of LLMs, his work is a great resource.



