import transformers

from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM
)

for model_name in ["bert-base-chinese"]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"assets/transformers/{model_name}")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.save_pretrained(f"assets/transformers/{model_name}")
