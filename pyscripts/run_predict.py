from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "seara/rubert-tiny2-russian-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

input_text = ["Привет, ты мне нравишься!", "Ах ты черт мерзкий"]
tokenized_text = tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    add_special_tokens=True,
)
outputs = model(**tokenized_text)
print(outputs.logits)
predicted = outputs.logits.softmax(-1)
print(predicted)
