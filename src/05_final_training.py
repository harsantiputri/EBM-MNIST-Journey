import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import random

# --- 1. Hyperparameters and Setup ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 100 # Let's train for longer to see the effects
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MCMC settings
CD_K = 60
MCMC_STEP_SIZE = 0.1
MCMC_NOISE = 0.1  # CRITICAL CHANGE: Increased noise to escape modes

# Loss settings
R1_COEFF = 1.0

# Anti-mode collapse setting
REINIT_FREQ = 0.1 # CRITICAL CHANGE: Increased frequency of re-initialization

OUTPUT_DIR = "EBM_004_Anti_Mode_Collapse_OUTPUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ... (ReplayBuffer, EBM, Data Loading, and MCMC function are unchanged) ...
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []
    def __len__(self): return len(self.buffer)
    def push(self, data):
        self.buffer.extend(data.detach().clone().cpu())
        self.buffer = self.buffer[-self.max_size:]
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return torch.stack(samples)

replay_buffer = ReplayBuffer()

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

class EBM(nn.Module):
    def __init__(self, input_size=784, hidden_size=1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x): return self.model(x)

def generate_samples_mcmc(model, x_init, k, step_size, noise_std):
    x_k = x_init.clone().detach().requires_grad_(True)
    for _ in range(k):
        if x_k.grad is not None: x_k.grad.zero_()
        energy = model(x_k).sum()
        grad = torch.autograd.grad(energy, x_k, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_k = x_k - step_size * grad + noise_std * torch.randn_like(x_k)
            x_k = torch.clamp(x_k, -1.0, 1.0)
        x_k = x_k.detach().requires_grad_(True)
    return x_k.detach()

# --- 5. Training Loop with Anti-Mode Collapse ---
model = EBM().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

print(f"Starting training on {DEVICE} with anti-mode collapse measures...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()

    for x_pos, _ in train_loader:
        x_pos = x_pos.to(DEVICE)

        # Positive phase and R1 Penalty
        x_pos.requires_grad_(True)
        energy_pos = model(x_pos)
        grad_pos = torch.autograd.grad(outputs=energy_pos.sum(), inputs=x_pos, create_graph=True)[0]
        grad_penalty = (grad_pos.view(grad_pos.shape[0], -1).norm(2, dim=1) ** 2).mean()
        x_pos = x_pos.detach()

        # Negative phase with increased re-initialization
        if len(replay_buffer) < BATCH_SIZE:
            x_neg_init = torch.rand_like(x_pos) * 2 - 1
        else:
            x_neg_init = replay_buffer.sample(BATCH_SIZE).to(DEVICE)
            # Use the new re-initialization frequency
            is_random = torch.rand(BATCH_SIZE, device=DEVICE) < REINIT_FREQ
            num_random = is_random.sum()
            if num_random > 0:
                 x_neg_init[is_random] = (torch.rand(num_random, x_pos.shape[1], device=DEVICE) * 2 - 1)

        x_neg = generate_samples_mcmc(model, x_neg_init, k=CD_K, step_size=MCMC_STEP_SIZE, noise_std=MCMC_NOISE)
        replay_buffer.push(x_neg)
        energy_neg = model(x_neg)

        # Loss Calculation
        loss_cd = energy_pos.mean() - energy_neg.mean()
        loss = loss_cd + R1_COEFF * grad_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Visualization
    if (epoch + 1) % 10 == 0: # Visualize every 10 epochs
        model.eval()
        noise = torch.randn(64, 784, device=DEVICE)
        # CRITICAL CHANGE: More steps for higher quality final samples
        samples = generate_samples_mcmc(model, noise, k=15000, step_size=MCMC_STEP_SIZE, noise_std=MCMC_NOISE)
        with torch.no_grad():
            samples = samples.view(-1, 1, 28, 28) * 0.5 + 0.5
            plt.figure(figsize=(8,8))
            for i in range(64):
                plt.subplot(8,8,i+1)
                plt.imshow(samples[i].cpu().squeeze(), cmap='gray')
                plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch+1}.png"))
            plt.close()

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "ebm_final.pth"))
print(f"Training finished. Model and samples saved in {OUTPUT_DIR}")