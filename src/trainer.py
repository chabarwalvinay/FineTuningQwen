from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


class FineTuningTrainer:
    def __init__(self, model, tokenizer, dataset, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.output_dir = output_dir

    def train(self):
        training_args = TrainingArguments(
            per_device_train_batch_size=6,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            learning_rate=2e-4,
            fp16=True,
            output_dir=self.output_dir,
            logging_steps=100,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            report_to=[],
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
