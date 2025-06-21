import torch
import numpy as np
import os
from train_latent_vectors import AutoencoderClassifier
from dataloader import get_dataloaders

MODEL_PATH = "models/autoencoder_classifier.pth"
OUTPUT_PATH = "models/latent_vectors_train.npz"
DATA_ROOT = "data/prepared_data"
BATCH_SIZE = 128
INPUT_DIM = 100
LATENT_DIM = 11
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Load model
    model = AutoencoderClassifier(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    # Get train loader
    train_loader, _ = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, shuffle=False)

    all_latents = []
    all_labels = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch['x'][:, :INPUT_DIM].to(DEVICE)
            # Use the same input as in training for latent extraction
            _, _, z = model(x)
            all_latents.append(z.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    np.savez(OUTPUT_PATH, latent=latents, label=labels)
    print(f"Saved latent vectors and labels to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
