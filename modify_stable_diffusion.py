# modify_stable_diffusion.py
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline

class CustomConditioningLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(CustomConditioningLayer, self).__init__()
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, embedding):
        embedding = self.relu(self.fc(embedding))
        conditioned_input = torch.cat((x, embedding.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        return conditioned_input

class ModifiedStableDiffusionModel(nn.Module):
    def __init__(self, original_model, conditioning_layer):
        super(ModifiedStableDiffusionModel, self).__init__()
        self.original_model = original_model
        self.conditioning_layer = conditioning_layer

    def forward(self, prompt, embedding):
        # Generate the initial embeddings from the prompt
        text_embeddings = self.original_model.text_encoder(prompt.input_ids)[0]

        # Integrate the custom conditioning layer
        conditioned_embeddings = self.conditioning_layer(text_embeddings, embedding)

        # Proceed with the rest of the generation process using the conditioned embeddings
        latents = self.original_model.vae.encode(conditioned_embeddings).latent_dist.sample()

        # Decoder part (adjust as necessary)
        latents = latents / self.original_model.vae.config.scaling_factor
        image = self.original_model.vae.decode(latents).sample()

        return image

def modify_stable_diffusion(model_id, embedding_dim, hidden_dim):
    # Load pre-trained Stable Diffusion model
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline.to("cuda")

    # Instantiate the custom conditioning layer
    conditioning_layer = CustomConditioningLayer(embedding_dim, hidden_dim)

    # Instantiate the modified Stable Diffusion model
    modified_model = ModifiedStableDiffusionModel(pipeline, conditioning_layer)

    return modified_model

if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"
    embedding_dim = 512  # Example embedding dimension
    hidden_dim = 768  # Hidden dimension of the model

    modified_model = modify_stable_diffusion(model_id, embedding_dim, hidden_dim)

    # Save the modified model
    torch.save(modified_model.state_dict(), "modified_stable_diffusion.pth")
