import torch
import numpy as np
from train_latent_vectors import Encoder, Decoder
from scipy.optimize import minimize

# --- Hyperparameters ---
INPUT_DIM = 100
LATENT_DIM = 11
ALPHA = 0.05  # Dissimilarity threshold
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ENCODER_PATH = "models/autoencoder_classifier.pth"
LATENT_DATA_PATH = "models/latent_vectors_train.npz"
DATA_ROOT = "data/prepared_data"
OUTPUT_PATH = "data/enhanced_data.npz"

def fermat_weber_point(latents):
    # Minimize sum of Euclidean distances to all points (Fermat-Weber)
    def objective(c):
        return np.sum(np.linalg.norm(latents - c, axis=1))
    init = np.mean(latents, axis=0)
    res = minimize(objective, init, method='Powell')
    return res.x

def dissimilarity(x, x_hat):
    return np.sqrt(np.mean((x - x_hat) ** 2))

def enhance_signal(x, encoder, decoder, anchor, alpha=ALPHA, max_steps=100):
    # Enhance signal in latent space, return enhanced latent vector
    x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        z = encoder(x_tensor)
        z = z.squeeze(0).cpu().numpy()
    dist = np.linalg.norm(z - anchor)
    m = int(np.ceil(dist / (alpha / LATENT_DIM)))
    m = min(m, max_steps)
    z_e = z.copy()
    for t in range(m + 1):
        z_t = (1 - t / m) * z + (t / m) * anchor
        z_t_tensor = torch.tensor(z_t, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        # Decoder will be applied later
        with torch.no_grad():
            x_hat = decoder(z_t_tensor).cpu().numpy().squeeze()
        if dissimilarity(x, x_hat) < alpha:
            z_e = z_t
            x_e = x_hat
        else:
            break
    return z_e, x_e

def main():
    # Load encoder and decoder
    model_state = torch.load(MODEL_ENCODER_PATH, map_location=DEVICE)
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
    print("Anchor (Fermat-Weber point) computed.")

    # Load all data
    from dataloader import get_dataloaders
    train_loader, _ = get_dataloaders(DATA_ROOT, batch_size=1, shuffle=False)
    enhanced_signals = []
    enhanced_labels = []
    original_signals = []

    for batch in train_loader:
        x = batch['x'][0, :INPUT_DIM].numpy()
        label = batch['label'][0]
        # Enhance signal in latent space and decode
        _, x_enhanced = enhance_signal(x, encoder, decoder, anchor, alpha=ALPHA)
        enhanced_signals.append(x_enhanced.astype(np.float32))
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
