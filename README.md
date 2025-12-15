
---

## ⚙️ Training Configuration
| Component | Value |
|--------|------|
Model | Qwen-0.6B Base
Fine-tuning | LoRA (PEFT)
Quantization | 4-bit NF4
Batch Size | 6
Grad Accumulation | 4
Epochs | 2
Optimizer | paged_adamw_8bit
Max Length | 512

---

## 📊 Results
- Training loss converged smoothly over 2 epochs
- Fine-tuned model showed improved instruction adherence
- LoRA adapters significantly reduced memory usage

> Only LoRA adapters were saved, not full model weights.

---

## 🧪 Inference
The repository includes example inference logic to compare:
- Base Qwen-0.6B
- LoRA fine-tuned model

This allows qualitative evaluation of instruction-following improvements.

---

## 🚫 Why Training Is Not Re-run Here
- Training requires long GPU runtime (5–6+ hours)
- Code is preserved for **reference, reproducibility, and review**
- Results were validated during original Kaggle execution

---

## 📌 Skills Demonstrated
- LLM fine-tuning & optimization
- Memory-efficient training strategies
- Clean Python OOP design
- ML experimentation → production-style refactor
