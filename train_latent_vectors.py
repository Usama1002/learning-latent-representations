import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import get_dataloaders
import os

MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


class Encoder(nn.Module):
    def __init__(self, input_dim=100, latent_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=11, output_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        self.scale = 1.5
    def forward(self, z):
        return self.scale * self.net(z)

class Classifier(nn.Module):
    def __init__(self, latent_dim=11):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)
        self.act = nn.Sigmoid()
    def forward(self, z):
        return self.act(self.fc(z)).squeeze(-1)

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim=100, latent_dim=11):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.classifier = Classifier(latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.classifier(z)
        return x_hat, y_hat, z

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#                                   TRAINING
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def train_model(
    root_dir="data/prepared_data",
    batch_size=128,
    input_dim=100,
    latent_dim=11,
    lr=1e-3,
    epochs=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Data
    train_loader, test_loader = get_dataloaders(root_dir, batch_size=batch_size)
    model = AutoencoderClassifier(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss, total_rec, total_cls = 0, 0, 0
        for batch in train_loader:
            x = batch['x'][:, :input_dim].to(device)  # Use only first 100 dims
            y = batch['y'][:, :input_dim].to(device)  # Use output waveform for reconstruction
            labels = batch['label'].float().to(device)
            optimizer.zero_grad()
            x_hat, y_hat, _ = model(y)
            rec_loss = mse_loss(x_hat, y)  # Use y as target for reconstruction
            # Selective gradient: only backprop BCE for valid samples (label==1)
            if (labels == 1).any():
                bce = bce_loss(y_hat[labels == 1], labels[labels == 1])
            else:
                bce = 0.0 * rec_loss  # No valid samples in batch
            loss = rec_loss - bce
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_rec += rec_loss.item() * y.size(0)
            total_cls += (bce.item() if isinstance(bce, torch.Tensor) else 0.0) * y.size(0)
        n = len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} | Recon: {total_rec/n:.4f} | BCE(valid): {total_cls/n:.4f}")

    # Optionally, save model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "autoencoder_classifier.pth"))
    print(f"Training complete. Model saved as {MODEL_SAVE_DIR}/autoencoder_classifier.pth.")

if __name__ == "__main__":
    train_model()
