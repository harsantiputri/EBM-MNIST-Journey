import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 1. Hyperparameters and Setup ---
LEARNING_RATE = 1e-5
BATCH_SIZE = 64
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MCMC settings
CD_K = 10
MCMC_STEP_SIZE = 0.01  # Further reduced for stability
MCMC_NOISE = 0.001

OUTPUT_DIR = "EBM_001_OUTPUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Model Definition ---
class EBM(nn.Module):
    def __init__(self, input_size=784, hidden_size=1024):  # Increased capacity
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)

# --- 4. Robust MCMC Sampling ---
def generate_samples_mcmc(model, x_init, k, step_size, noise_std):
    """Completely revised sampling with proper gradient handling"""
    x_k = x_init.clone().detach()
    x_k.requires_grad_(True)
    
    for _ in range(k):
        # Zero gradients
        if x_k.grad is not None:
            x_k.grad.zero_()
        
        # Forward pass
        energy = model(x_k).sum()
        
        # Backward pass
        grad = torch.autograd.grad(energy, x_k, retain_graph=False, create_graph=False)[0]
        
        # Update with noise - detach from computation graph
        with torch.no_grad():
            x_k = x_k - step_size * grad + noise_std * torch.randn_like(x_k)
            x_k = torch.clamp(x_k, -1.0, 1.0)
            
        # Re-enable gradients for next iteration
        x_k = x_k.detach().requires_grad_(True)
    
    return x_k.detach()

# --- 5. Training Loop with Stability Improvements ---
model = EBM().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))  # More stable optimizer params

for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()
    
    for x_pos, _ in train_loader:
        x_pos = x_pos.to(DEVICE)
        
        # Positive phase
        energy_pos = model(x_pos)
        
        # Negative phase - start from uniform noise
        x_neg_init = torch.rand_like(x_pos) * 2 - 1  # Uniform [-1, 1]
        x_neg = generate_samples_mcmc(model, x_neg_init, k=CD_K,
                                    step_size=MCMC_STEP_SIZE,
                                    noise_std=MCMC_NOISE)
        energy_neg = model(x_neg)
        
        # Loss with gradient penalty for stability
        loss = energy_pos.mean() - energy_neg.mean()
        
        # Gradient penalty
        gp = (energy_pos**2).mean() + (energy_neg**2).mean()
        loss = loss + 0.5 * gp
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # --- Corrected Visualization Block ---

if (epoch + 1) % 5 == 0:
    model.eval()

    # --- Step 1: Generate samples (requires gradients) ---
    noise = torch.randn(64, 784, device=DEVICE)
    samples = generate_samples_mcmc(model, noise, k=500,
                                    step_size=MCMC_STEP_SIZE*2, # Using slightly larger steps for generation
                                    noise_std=MCMC_NOISE*2)

    # --- Step 2: Process and plot samples (does NOT require gradients) ---
    with torch.no_grad():
        # Move tensor processing and plotting into the no_grad block for efficiency
        samples = samples.view(-1, 1, 28, 28) * 0.5 + 0.5 # Denormalize

        plt.figure(figsize=(8,8))
        for i in range(64):
            plt.subplot(8,8,i+1)
            plt.imshow(samples[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
        
        # Corrected save path
        plt.savefig(os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch+1}.png"))
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "ebm_final.pth"))