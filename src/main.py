"""
Main training pipeline.

NOTE:
Actual training was performed on Kaggle using GPU.
This script demonstrates the full pipeline but is not
intended to be executed locally.
"""

from utils import setup_environment
from model import QwenLoRAModel
from dataset import AlpacaDataset
from trainer import FineTuningTrainer

MODEL_PATH = "Qwen-0.6B-Base"
DATA_PATH = "alpaca_data.json"
OUTPUT_DIR = "qwen_lora_output"


def main():
    setup_environment()

    model_builder = QwenLoRAModel(MODEL_PATH)
    tokenizer = model_builder.load_tokenizer()
    model = model_builder.load_model()

    dataset = AlpacaDataset(
        data_path=DATA_PATH, tokenizer=tokenizer, max_length=512
    ).load()

    trainer = FineTuningTrainer(
        model=model, tokenizer=tokenizer, dataset=dataset, output_dir=OUTPUT_DIR
    )

    # trainer.train()  # Uncomment only if you want to retrain


if __name__ == "__main__":
    main()
