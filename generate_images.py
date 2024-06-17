# generate_images.py
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import pickle

from modify_stable_diffusion import ModifiedStableDiffusionModel, CustomConditioningLayer

# Function to get text embedding from CLIP
def get_text_embedding(clip_tokenizer, clip_model, prompt):
    inputs = clip_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        text_embeddings = clip_model(**inputs).last_hidden_state
    return text_embeddings

# Load character embeddings
with open('character_embeddings.pkl', 'rb') as f:
    character_embeddings = pickle.load(f)

# Load the modified Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline.to("cuda")

embedding_dim = 512  # Example embedding dimension
hidden_dim = 768  # Hidden dimension of the model

conditioning_layer = CustomConditioningLayer(embedding_dim, hidden_dim)
modified_model = ModifiedStableDiffusionModel(pipeline, conditioning_layer)
modified_model.load_state_dict(torch.load("modified_stable_diffusion.pth"))
modified_model.to("cuda")

clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Generate image function
def generate_image_with_custom_conditioning(prompt, character_id):
    # Retrieve the character embedding
    embedding = character_embeddings.get(character_id)
    embedding = torch.tensor(embedding).unsqueeze(0).to("cuda")

    # Get text embedding
    text_embedding = get_text_embedding(clip_tokenizer, clip_model, prompt)

    # Generate the image using the modified model
    image = modified_model(text_embedding, embedding)

    return image

# Example usage
if __name__ == "__main__":
    prompt = "A brave knight standing in a battlefield"
    character_id = "character_id_1"
    generated_image = generate_image_with_custom_conditioning(prompt, character_id)
    generated_image.save("generated_image_with_custom_conditioning.png")
