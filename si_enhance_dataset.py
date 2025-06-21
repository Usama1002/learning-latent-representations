import torch
import numpy as np
import os
from train_latent_vectors import Encoder, Decoder
from si_enhancement import fermat_weber_point, enhance_signal
from dataloader import get_dataloaders
from tqdm import tqdm

INPUT_DIM = 100
LATENT_DIM = 11
ALPHA = 0.05
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "models/autoencoder_classifier.pth"
LATENT_DATA_PATH = "models/latent_vectors_train.npz"
DATA_ROOT = "data/prepared_data"
OUTPUT_PATH = "data/enhanced_data.npz"

def main():
    # Load encoder and decoder
    model_state = torch.load(MODEL_PATH, map_location=DEVICE)
    encoder = Encoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
    decoder = Decoder(LATENT_DIM, INPUT_DIM).to(DEVICE)
    encoder.load_state_dict(model_state['encoder'] if 'encoder' in model_state else {k.replace('encoder.', ''): v for k, v in model_state.items() if k.startswith('encoder.')})
    decoder.load_state_dict(model_state['decoder'] if 'decoder' in model_state else {k.replace('decoder.', ''): v for k, v in model_state.items() if k.startswith('decoder.')})
    encoder.eval()
    decoder.eval()

    # Load latent vectors and labels for anchor computation
    data = np.load(LATENT_DATA_PATH)
    latents = data['latent']
    labels = data['label']
    valid_latents = latents[labels == 1]
    anchor = fermat_weber_point(valid_latents)

    # Load all data
    train_loader, _ = get_dataloaders(DATA_ROOT, batch_size=1, shuffle=False)
    enhanced_signals = []
    enhanced_labels = []
    original_signals = []

    for batch in tqdm(train_loader, desc="Enhancing signals"):
        x = batch['x'][0, :INPUT_DIM].numpy()
        label = batch['label'][0]
        # Enhance signal in latent space and decode
        x_enhanced_latent = enhance_signal(x, encoder, decoder, anchor, alpha=ALPHA)
        # Optionally, re-encode and decode for consistency, or just use x_enhanced_latent as enhanced signal
        enhanced_signals.append(x_enhanced_latent.astype(np.float32))
        enhanced_labels.append(label)
        original_signals.append(x.astype(np.float32))

    enhanced_signals = np.stack(enhanced_signals)
    enhanced_labels = np.array(enhanced_labels)
    original_signals = np.stack(original_signals)

    # Save in similar format as prepared data
    np.savez(OUTPUT_PATH, x=original_signals, x_enhanced=enhanced_signals, label=enhanced_labels)
    print(f"Enhanced data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
