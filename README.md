# eQ-DIMON: Enhanced Quantum Deep Implicit Operator Network

This repository implements **eQ-DIMON**, a hybrid framework combining classical deep learning and quantum computing to solve partial differential equations (PDEs) on diffeomorphic domains. Specifically, it targets the Laplace equation with parametric boundary conditions and domain deformations.

## Overview

The eQ-DIMON framework integrates an **EnhancedMIONet** neural network with a quantum layer to model PDE solutions efficiently. It uses PyTorch for the classical components and PennyLane for the quantum circuit, leveraging GPU acceleration when available. The program generates training data, trains the model, and evaluates its performance on test samples, with visualization of results.

### Key Features
- Solves the Laplace equation on parametric domains using diffeomorphisms.
- Combines a multi-branch neural network (for domain parameters, boundary conditions, and spatial coordinates) with a 2-qubit quantum circuit.
- Implements a hybrid loss function including MSE, PDE residuals, boundary condition enforcement, and quantum regularization.
- Uses multiprocessing for efficient training data generation.
- Supports training with early stopping and validation split.
- Visualizes true vs. predicted solutions and training metrics.

## Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `torch`
  - `pennylane`
  - `scipy`
  - `matplotlib`
- Optional: CUDA-enabled GPU for faster computation

Install dependencies using:

```bash
pip install numpy torch pennylane scipy matplotlib
```

## Usage

### Clone the Repository

```bash
git clone https://github.com/jfinst1/eQ_DIMON
cd eq-dimon
```

### Run the Program

Execute the main script to generate data, train the model, and visualize results:

```bash
python main.py
```

- Generates 300 training samples.
- Trains for 5 epochs (configurable via `epochs` parameter).
- Tests on 10 random samples and displays plots.

### Output
- Console logs with training progress, test MSEs, and runtime.
- A Matplotlib figure showing:
  - True solution
  - Predicted solution
  - Training and validation loss/MSE curves

## Code Structure

- **`main.py`**:
  - **PDE Problem**: Defines domain generation and numerical Laplace solver.
  - **EnhancedMIONet**: Neural network with quantum layer.
  - **eQ_DIMON**: Main class for training and prediction.
  - **Data Generation**: Multiprocessing-based training data creation.
  - **Main Execution**: Orchestrates data generation, training, testing, and plotting.

## Example Output

After running, expect:

- Average test MSE over 10 samples (e.g., `0.001234`).
- A plot comparing true and predicted solutions for a test sample, alongside training metrics.

## Customization

- **Training Parameters**: Adjust `n_samples`, `epochs`, `batch_size`, `initial_lr`, etc., in the `eQ_DIMON` initialization or train call.
- **Domain Parameters**: Modify `theta` and boundary condition (`bc`) ranges in `generate_training_data`.
- **Network Architecture**: Tweak `hidden_dim` or `num_quantum_weights` in `EnhancedMIONet`.

## License

This project is licensed under my craziness. See the `LICENSE` file for details.

## Acknowledgments

- Built with PyTorch and PennyLane.
- Inspired by advances in physics-informed neural networks and quantum machine learning.
