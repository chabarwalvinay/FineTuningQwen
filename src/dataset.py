from datasets import load_dataset


class AlpacaDataset:
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _format_example(self, example):
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )

        tokenized = self.tokenizer(
            prompt, truncation=True, padding="max_length", max_length=self.max_length
        )

        return {"input_ids": tokenized["input_ids"], "labels": tokenized["input_ids"]}

    def load(self):
        dataset = load_dataset("json", data_files=self.data_path)["train"]

        dataset = dataset.map(self._format_example, remove_columns=dataset.column_names)

        return dataset
