import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# Initialize FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()  # Load pretrained model

# Define function to generate face embeddings
def generate_face_embedding(image_path):
    img = Image.open(image_path)
    img = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img)[0].numpy()
    return embedding

# Example characters and their image paths
character_images = {
    "character_id_1": "images\character_id_1.jpg",
    # Add more characters as needed
}

# Generate embeddings for each character
character_embeddings = {}
for character_id, image_path in character_images.items():
    embedding = generate_face_embedding(image_path)
    character_embeddings[character_id] = embedding

# Save embeddings for later use
with open('embeddings\character_embeddings.pkl', 'wb') as f:
    pickle.dump(character_embeddings, f)
