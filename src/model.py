import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType


class QwenLoRAModel:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return tokenizer

    def load_model(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, quantization_config=quant_config, device_map="auto"
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "q_proj", "k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        return model
