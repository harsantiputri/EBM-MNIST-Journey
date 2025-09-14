import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

# --- 1. Define the Environment and Model Architecture ---
# These match the definitions from our training script.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "EBM_004_High_Fidelity_Generation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Re-define the EBM class
class EBM(nn.Module):
    def __init__(self, input_size=784, hidden_size=1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x): return self.model(x)

# Re-define the MCMC sampling function
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


# --- 2. Load the Trained Model ---

# Path to our saved model from the previous run
PATH_TO_MODEL = "EBM_004_Anti_Mode_Collapse_OUTPUT/ebm_final.pth"

# Instantiate the model architecture
model = EBM().to(DEVICE)

# Load the saved weights
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=DEVICE))

# Set the model to evaluation mode.
# It disables dropout or batch normalization if they were used.
model.eval()

print(f"Model loaded from {PATH_TO_MODEL} and set to evaluation mode.")


# --- 3. Generate High-Fidelity Samples ---

# Generation Hyperparameters
NUM_SAMPLES = 64
MCMC_K = 75000          # <-- Dramatically increased number of steps
MCMC_STEP_SIZE = 0.1    # We can experiment with this
MCMC_NOISE = 0.05       # And this. Sometimes reducing noise at the end helps.

# Start generation from random noise
initial_noise = torch.randn(NUM_SAMPLES, 784, device=DEVICE)

print(f"Starting high-fidelity generation with k={MCMC_K} steps...")
# The MCMC process requires gradients, so we do this outside a no_grad() block
samples = generate_samples_mcmc(model, initial_noise, k=MCMC_K, step_size=MCMC_STEP_SIZE, noise_std=MCMC_NOISE)
print("Generation complete.")


# --- 4. Save the Generated Images ---

# Plotting does not require gradients
with torch.no_grad():
    samples = samples.view(-1, 1, 28, 28) * 0.5 + 0.5 # Denormalize
    plt.figure(figsize=(8,8))
    for i in range(NUM_SAMPLES):
        plt.subplot(8,8,i+1)
        plt.imshow(samples[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, f"high_fidelity_samples_k{MCMC_K}.png")
    plt.savefig(save_path)
    plt.close()

print(f"High-fidelity samples saved to {save_path}")
