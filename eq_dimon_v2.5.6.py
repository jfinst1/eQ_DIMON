import numpy as np
import torch
import torch.nn as nn
from torch.func import grad, vmap
import pennylane as qml
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time
import logging
import json
import os
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds and device
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Initializing with device: {device}")

# --- Step 1: PDE Problem and Diffeomorphic Domains ---
def generate_parametric_domain(theta, N=64):
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    a1, a2, a3 = theta
    x = X + a1 * np.cos(a2 * np.pi + a3) * np.sin(a2 * np.pi + a3)
    y = Y + a1 * np.sin(a2 * np.pi + a3)**2
    return x, y

def reference_domain(N=64):
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    return np.stack([X, Y], axis=-1)

def subsampled_reference_domain(N=8):
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    return np.stack([X, Y], axis=-1)

def diffeomorphism(x, y, theta):
    a1, a2, a3 = theta
    X = x - a1 * np.cos(a2 * np.pi + a3) * np.sin(a2 * np.pi + a3)
    Y = y - a1 * np.sin(a2 * np.pi + a3)**2
    return X, Y

def solve_laplace(theta, bc, N=64):
    x, y = generate_parametric_domain(theta, N)
    u = np.zeros((N, N))
    u[0, :] = bc[0]
    u[-1, :] = bc[1]
    u[:, 0] = bc[2]
    u[:, -1] = bc[3]
    for _ in range(2000):
        u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
    return u

# --- Step 2: EnhancedMIONet with Quantum Layer ---
class EnhancedMIONet(nn.Module):
    def __init__(self, theta_dim=3, bc_dim=4, hidden_dim=512, num_quantum_weights=6):
        super(EnhancedMIONet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.branch_theta = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.branch_bc = nn.Sequential(
            nn.Linear(bc_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.num_quantum_weights = num_quantum_weights
        self.quantum_weights = nn.Parameter(torch.randn(num_quantum_weights, device=self.device, dtype=torch.float32) * 0.1)
        torch_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.quantum_dev = qml.device("default.qubit.torch", wires=2, torch_device=torch_device_str)
        
        @qml.qnode(self.quantum_dev, interface='torch')
        def quantum_circuit(inputs, weights):
            inputs = torch.pi * (inputs - inputs.min(dim=1, keepdim=True)[0]) / \
                     (inputs.max(dim=1, keepdim=True)[0] - inputs.min(dim=1, keepdim=True)[0] + 1e-8)
            for i in range(6):
                qml.RY(inputs[..., i], wires=i % 2)
                qml.RX(weights[i], wires=i % 2)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
        
        self.quantum_circuit = quantum_circuit
    
    def quantum_layer(self, inputs):
        z0, z1 = self.quantum_circuit(inputs, self.quantum_weights)
        return ((z0 + z1) / 2).to(dtype=torch.float32)
    
    def forward(self, theta, bc, X_ref):
        batch_size = theta.shape[0]
        n_points = X_ref.shape[-2]  # Handles [batch, points, 2] or [points, 2]
        
        theta_out = self.branch_theta(theta)  # [batch, 512] or [batch, 1, 512]
        bc_out = self.branch_bc(bc)          # [batch, 512] or [batch, 1, 512]
        trunk_out = self.trunk(X_ref)        # [batch, points, 512] or [points, 512]
        
        # Squeeze extra dims if present
        if theta_out.dim() == 3:
            theta_out = theta_out.squeeze(1)  # [batch, 512]
        if bc_out.dim() == 3:
            bc_out = bc_out.squeeze(1)        # [batch, 512]
        
        # Expand to match points
        theta_out = theta_out.unsqueeze(1).expand(-1, n_points, -1)  # [batch, points, 512]
        bc_out = bc_out.unsqueeze(1).expand(-1, n_points, -1)        # [batch, points, 512]
        
        # Adjust trunk_out if needed
        if trunk_out.dim() == 2:
            trunk_out = trunk_out.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, points, 512]
        
        combined = theta_out * bc_out * trunk_out
        quantum_input = torch.cat((theta, bc[..., :3]), dim=-1)  # [batch, 6]
        quantum_output = self.quantum_layer(quantum_input)       # [batch]
        final_output = self.final_layer(combined) * (1 + quantum_output.unsqueeze(-1).unsqueeze(-1))
        logging.debug(f"Forward: combined_shape={combined.shape}, quantum_shape={quantum_output.shape}, final_shape={final_output.shape}")
        return final_output

# --- Step 4: eQ-DIMON Hybrid Framework ---
class eQ_DIMON:
    def __init__(self, batch_size=64, initial_lr=0.001, weight_decay=1e-4, quantum_weight=1.0):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedMIONet().to(self.device)
        self.batch_size = batch_size  # Locked at 64
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.quantum_weight = quantum_weight
        logging.info(f"eQ-DIMON initialized: batch_size={batch_size}, initial_lr={initial_lr}, weight_decay={weight_decay}, quantum_weight={quantum_weight}")

    def compute_pde_residual(self, theta_batch, bc_batch, X_ref_sub):
        batch_size = theta_batch.shape[0]  # Should be 64
        X_ref_tensor = torch.tensor(X_ref_sub.reshape(-1, 2), dtype=torch.float32, device=self.device, requires_grad=True)  # [64, 2]
        theta_tensor = torch.tensor(theta_batch, dtype=torch.float32, device=self.device)
        bc_tensor = torch.tensor(bc_batch, dtype=torch.float32, device=self.device)
        
        # Forward pass
        u = self.model(theta_tensor, bc_tensor, X_ref_tensor).squeeze(-1)  # [batch, 64]
        
        # First derivatives
        grad_u = torch.autograd.grad(u.sum(), X_ref_tensor, create_graph=True, retain_graph=True)[0]  # [64, 2]
        u_x = grad_u[:, 0].view(batch_size, -1)  # [batch, 64]
        u_y = grad_u[:, 1].view(batch_size, -1)  # [batch, 64]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), X_ref_tensor, create_graph=False, retain_graph=True)[0][:, 0].view(batch_size, -1)  # [batch, 64]
        u_yy = torch.autograd.grad(u_y.sum(), X_ref_tensor, create_graph=False)[0][:, 1].view(batch_size, -1)  # [batch, 64]
        
        laplacian = u_xx + u_yy  # [batch, 64]
        residual = torch.mean(laplacian**2)
        
        logging.debug(f"PDE: u_mean={u.mean().item():.6f}, u_x_mean={u_x.mean().item():.6f}, "
                      f"u_xx_mean={u_xx.mean().item():.6f}, u_yy_mean={u_yy.mean().item():.6f}, "
                      f"residual={residual.item():.6f}")
        torch.cuda.empty_cache()
        return residual

    def _train_batch(self, batch, X_ref_sub, X_ref_full):
        theta_batch, bc_batch, u_batch = batch  # Unpack as tensors
        theta_batch = theta_batch.to(self.device)
        bc_batch = bc_batch.to(self.device)
        u_batch = u_batch.to(self.device).view(-1, 4096)  # Flatten to [batch, 4096]
        X_ref_full_tensor = torch.tensor(X_ref_full.reshape(-1, 2), dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        u_pred = self.model(theta_batch, bc_batch, X_ref_full_tensor).squeeze(-1)  # [batch, 4096]
        mse_loss = torch.mean((u_pred - u_batch) ** 2)
        
        pde_loss = self.compute_pde_residual(theta_batch, bc_batch, X_ref_sub)
        
        bc_pred = u_pred[:, [0, 63, 4032, 4095]]  # Top, bottom, left, right indices for 64x64
        bc_loss = torch.mean((bc_pred - bc_batch) ** 2)
        
        quantum_params = torch.cat([p.flatten() for p in self.model.quantum_weights])
        quantum_penalty = torch.mean((quantum_params - 0.5) ** 2)
        
        loss = mse_loss + 10.0 * pde_loss + bc_loss + self.quantum_weight * quantum_penalty
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), mse_loss.item(), pde_loss.item(), bc_loss.item(), quantum_penalty.item()

    def train(self, data, epochs=500, patience=20, val_split=0.2):
        # Unpack and convert to tensors upfront
        theta_data, bc_data, _, u_data = zip(*data)  # Ignore X_ref
        theta_data = torch.tensor(np.stack(theta_data), dtype=torch.float32)
        bc_data = torch.tensor(np.stack(bc_data), dtype=torch.float32)
        u_data = torch.tensor(np.stack(u_data), dtype=torch.float32)
        
        n_samples = len(data)
        n_val = int(val_split * n_samples)
        n_train = n_samples - n_val
        n_train_batches = n_train // self.batch_size
        n_train = n_train_batches * self.batch_size  # Drop incomplete batch
        
        perm = torch.randperm(n_samples)
        train_idx, val_idx = perm[:n_train], perm[n_train:n_train + n_val]
        
        train_theta = theta_data[train_idx]
        train_bc = bc_data[train_idx]
        train_u = u_data[train_idx]
        val_theta = theta_data[val_idx]
        val_bc = bc_data[val_idx]
        val_u = u_data[val_idx]
        
        logging.info(f"Data split: {n_train} training samples, {len(val_idx)} validation samples")
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, train_mses, val_losses, val_mses = [], [], [], []  # Fixed: four lists
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss, epoch_mse = 0.0, 0.0
            for i in range(0, n_train, self.batch_size):
                batch_theta = train_theta[i:i + self.batch_size]
                batch_bc = train_bc[i:i + self.batch_size]
                batch_u = train_u[i:i + self.batch_size]
                batch = (batch_theta, batch_bc, batch_u)
                batch_loss, mse, pde, bc, quantum = self._train_batch(batch, X_ref_sub, X_ref_full)
                epoch_loss += batch_loss
                epoch_mse += mse
            
            epoch_loss /= n_train_batches
            epoch_mse /= n_train_batches
            train_losses.append(epoch_loss)
            train_mses.append(epoch_mse)
            
            self.model.eval()
            with torch.no_grad():
                val_loss, val_mse = 0.0, 0.0
                n_val_batches = len(val_idx) // self.batch_size
                for i in range(0, n_val_batches * self.batch_size, self.batch_size):
                    batch_theta = val_theta[i:i + self.batch_size]
                    batch_bc = val_bc[i:i + self.batch_size]
                    batch_u = val_u[i:i + self.batch_size]
                    batch = (batch_theta, batch_bc, batch_u)
                    X_ref_full_tensor = torch.tensor(X_ref_full.reshape(-1, 2), dtype=torch.float32, device=self.device)
                    u_pred = self.model(batch_theta.to(self.device), batch_bc.to(self.device), X_ref_full_tensor).squeeze(-1)  # [batch, 4096]
                    u_batch_flat = batch_u.to(self.device).view(-1, 4096)  # Flatten to [batch, 4096]
                    val_mse += torch.mean((u_pred - u_batch_flat) ** 2).item()
                    pde_loss = self.compute_pde_residual(batch_theta, batch_bc, X_ref_sub)
                    bc_pred = u_pred[:, [0, 63, 4032, 4095]]
                    bc_loss = torch.mean((bc_pred - batch_bc.to(self.device)) ** 2).item()
                    quantum_params = torch.cat([p.flatten() for p in self.model.quantum_weights])
                    quantum_penalty = torch.mean((quantum_params - 0.5) ** 2).item()
                    val_loss += val_mse + 10.0 * pde_loss + bc_loss + self.quantum_weight * quantum_penalty
            
            val_loss /= n_val_batches if n_val_batches > 0 else 1
            val_mse /= n_val_batches if n_val_batches > 0 else 1
            val_losses.append(val_loss)
            val_mses.append(val_mse)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
        
        return train_losses, train_mses, val_losses, val_mses

    def predict(self, theta, bc, X_ref):
        theta_tensor = torch.tensor(theta, dtype=torch.float32, device=self.device).unsqueeze(0)
        bc_tensor = torch.tensor(bc, dtype=torch.float32, device=self.device).unsqueeze(0)
        X_flat = torch.tensor(X_ref.reshape(-1, 2), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            u_pred = self.model(theta_tensor, bc_tensor, X_flat).squeeze().cpu().numpy()
        
        u_pred = u_pred.reshape(X_ref.shape[0], X_ref.shape[1])
        u_pred[0, :] = bc[0]
        u_pred[-1, :] = bc[1]
        u_pred[:, 0] = bc[2]
        u_pred[:, -1] = bc[3]
        return u_pred

# --- Step 5: Data Generation and Benchmarking ---
def generate_training_data_worker(args):
    theta, bc = args
    X_ref = reference_domain()
    u = solve_laplace(theta, bc)
    X_mapped, Y_mapped = diffeomorphism(*generate_parametric_domain(theta), theta)
    u_ref = griddata((X_mapped.flatten(), Y_mapped.flatten()), u.flatten(), (X_ref[..., 0], X_ref[..., 1]), method='cubic')
    return (theta, bc, X_ref, u_ref)

def generate_training_data(n_samples=300):
    logging.info(f"Generating {n_samples} training samples")
    pool = mp.Pool(mp.cpu_count())
    thetas = np.random.uniform([-0.5, 0, -0.5], [0.5, 0.75, 0.5], (n_samples, 3))
    bcs = np.random.uniform(0, 1, (n_samples, 4))
    data = pool.map(generate_training_data_worker, zip(thetas, bcs))
    pool.close()
    logging.info(f"Training data generated: {len(data)} samples")
    return data

# --- Main Execution ---
if __name__ == '__main__':
    start_time = time()
    start = time()
    logging.info("Generating training data...")
    data = generate_training_data(n_samples=300)
    
    logging.info("Training eQ-DIMON...")
    eq_dimon = eQ_DIMON(batch_size=64)
    X_ref_sub = subsampled_reference_domain(N=8)  # 8x8 = 64 points
    X_ref_full = reference_domain()
    losses, mses, val_losses, val_mses = eq_dimon.train(data, epochs=5, patience=20, val_split=0.2)
    print(f"5 epochs took {time() - start:.2f} seconds")
    
    logging.info("Testing on 10 random samples...")
    test_indices = np.random.choice(len(data), 10, replace=False)
    test_mses = []
    X_ref = reference_domain()
    for idx in test_indices:
        theta, bc, _, u_true = data[idx]
        u_pred = eq_dimon.predict(theta, bc, X_ref)
        mse = np.mean((u_pred - u_true)**2)
        test_mses.append(mse)
        logging.info(f"Test sample {idx}: theta={theta}, bc={bc}, MSE={mse:.6f}")
    
    avg_test_mse = np.mean(test_mses)
    logging.info(f"Average Test MSE over 10 samples: {avg_test_mse:.6f}")
    
    theta_test, bc_test, _, u_true = data[test_indices[0]]
    u_pred = eq_dimon.predict(theta_test, bc_test, X_ref)
    x_test, y_test = generate_parametric_domain(theta_test)
    logging.info("Plotting results for first test sample")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.contourf(x_test, y_test, u_true, levels=20)
    plt.title("True Solution")
    plt.subplot(1, 3, 2)
    plt.contourf(x_test, y_test, griddata((X_ref[..., 0].flatten(), X_ref[..., 1].flatten()), u_pred.flatten(), (x_test, y_test), method='cubic'), levels=20)
    plt.title("Predicted Solution")
    plt.subplot(1, 3, 3)
    plt.plot(losses, label='Train Loss')
    plt.plot(mses, label='Train MSE')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(val_mses, label='Val MSE')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training Metrics")
    plt.show()
    
    logging.info(f"Total runtime: {time() - start_time:.2f} seconds")
    logging.info("Program execution completed")