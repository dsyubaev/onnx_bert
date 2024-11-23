from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_checkpoint = "seara/rubert-tiny2-russian-sentiment"

save_directory = "./data/rubert-tiny2-russian-sentiment-onnx"

# Load a model from transformers and export it to ONNX
ort_model = ORTModelForSequenceClassification.from_pretrained(
    model_checkpoint, export=True
)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Save the onnx model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
