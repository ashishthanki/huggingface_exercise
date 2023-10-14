"File for practicing hugging face transformers image classifiers"
from transformers import pipeline

# use model for image classification
classifier = pipeline(model="google/vit-base-patch16-224")

black_bear_url = "https://www.wildlifeillinois.org/wp-content/uploads/2019/01/Black-bear.jpeg"
cat_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

preds = classifier(images=[black_bear_url, cat_url])


for results in preds:
    top_pred = results[0] # retrieve highest score label prediction
    print(f"Labels: {top_pred['label']} \n. Confidence: {top_pred['score']}")
print(preds)