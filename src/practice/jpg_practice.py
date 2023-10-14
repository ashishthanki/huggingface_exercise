import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

black_bear_url = 'https://i.natgeofe.com/k/caf9703e-c4e2-44e6-8b82-2c8b480cacc8/black-bear-cubs_4x3.jpg'
image = Image.open(requests.get(black_bear_url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])