
# initialization
from transformers import CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

def predict_clip(image_file_path='/Users/evgenia/OpenEyesClassificator/eyes/closed/000003.jpg'):
    image = clip_preprocess(Image.open(image_file_path)).unsqueeze(0).to(device)
    clip_model, _ = clip.load('ViT-B/32', device)

    # Calculate features
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    predictions = {}
    for value, index in zip(values, indices):
        predictions[f"{categories[index]:>16s}"] = f"{100 * value.item():.4f}%"
    return predictions

predict_clip()