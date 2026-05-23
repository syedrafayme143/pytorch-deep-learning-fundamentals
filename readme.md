# 🔥 PyTorch Deep Learning — From Fundamentals to CNNs

> A structured, hands-on implementation of core Deep Learning concepts using PyTorch — covering everything from tensor operations and autograd to convolutional networks and transfer learning.

---

## 🗂️ Repository Structure

```
pytorch-deep-learning-fundamentals/
│
├── 02_tensor_basics.py            # Tensor creation, ops, NumPy bridge, GPU support
├── 03_autograd.py                 # Automatic differentiation & computation graphs
├── 04_backpropagation.py          # Manual backprop with autograd engine
├── 05_1_gradientdescent_manually.py  # Gradient descent from scratch with NumPy
├── 05_2_gradientdescent_auto.py   # Gradient descent using torch.autograd
├── 06_1_loss_and_optimizer.py     # nn.MSELoss + SGD optimizer
├── 06_2_model_loss_and_optimizer.py  # Full model pipeline using nn.Linear
├── 07_linear_regression.py        # End-to-end linear regression with sklearn data
├── 08_logistic_regression.py      # Binary classification on Breast Cancer dataset
├── 09_dataloader.py               # Custom Dataset class + DataLoader batching
├── 10_transformers.py             # Custom data transforms & torchvision.Compose
├── 11_softmax_and_crossentropy.py # Softmax, NLLLoss, CrossEntropyLoss
├── 12_activation_functions.py     # ReLU, Sigmoid, Tanh, Leaky ReLU
├── 12_plot_activations.py         # Visualization of activation functions
├── 13_feedforward.py              # Fully-connected NN on MNIST
├── 14_cnn.py                      # ConvNet on CIFAR-10 with per-class accuracy
├── 15_transfer_learning.py        # ResNet-18 fine-tuning & feature extraction
├── 16_tensorboard.py              # TensorBoard logging (loss, accuracy, PR curves)
├── 17_save_load.py                # Model persistence: state_dict & checkpointing
│
├── requirements.txt
└── README.md
```

---

## 🧠 Concepts Covered

### 1. Tensor Operations (`02_tensor_basics.py`)
Covers tensor initialization strategies (`torch.empty`, `torch.rand`, `torch.zeros`), dtype specification, in-place vs. out-of-place operations, slicing, reshaping with `view()`, and the **shared-memory NumPy bridge**. Includes GPU tensor creation via `torch.device("cuda")`.

### 2. Autograd & Computation Graphs (`03_autograd.py`)
Demonstrates PyTorch's dynamic computation graph via `requires_grad=True`. Covers `grad_fn` attributes, the vector-Jacobian product for non-scalar outputs, and three methods for detaching tensors from the graph: `.requires_grad_(False)`, `.detach()`, and `torch.no_grad()`. Highlights the critical need for `grad.zero_()` to prevent gradient accumulation across iterations.

### 3. Backpropagation (`04_backpropagation.py`)
Implements the full forward → loss → `.backward()` → weight update cycle for a single-parameter model, demonstrating how `loss.backward()` populates `.grad` and how weight updates are applied outside the computation graph using `torch.no_grad()`.

### 4. Gradient Descent — Manual vs. Autograd (`05_1`, `05_2`)
Side-by-side comparison of gradient computation: `05_1` derives and applies the MSE gradient analytically using NumPy; `05_2` replaces the manual gradient with PyTorch's autograd, illustrating the exact substitution point and why `w.grad.zero_()` is essential.

### 5. Loss Functions & Optimizers (`06_1`, `06_2`)
Introduces the standard PyTorch training recipe: `nn.MSELoss` for regression, `torch.optim.SGD`, and the `optimizer.step()` / `optimizer.zero_grad()` loop. `06_2` promotes the weight `w` to a full `nn.Linear` module, showing how `model.parameters()` feeds the optimizer.

### 6. Linear Regression (`07_linear_regression.py`)
Full pipeline using `sklearn.datasets.make_regression`, numpy-to-tensor casting, `nn.Linear`, MSE loss, and SGD. Includes matplotlib visualization of the fitted regression line against raw data.

### 7. Logistic Regression (`08_logistic_regression.py`)
Binary classification on the **Wisconsin Breast Cancer** dataset. Implements a custom `nn.Module` with a sigmoid output layer, `nn.BCELoss`, feature standardization with `StandardScaler`, and reports test accuracy using `torch.no_grad()`.

### 8. DataLoader & Custom Datasets (`09_dataloader.py`)
Implements the `Dataset` interface (`__init__`, `__getitem__`, `__len__`) for a CSV-based Wine dataset. Demonstrates `DataLoader` with batching, shuffling, and multi-worker loading. Covers epoch/batch/iteration terminology and built-in `torchvision.datasets.MNIST`.

### 9. Transforms & Pipelines (`10_transformers.py`)
Builds a reusable transform pipeline with custom `ToTensor` and `MulTransform` classes, composed via `torchvision.transforms.Compose`. Demonstrates the clean separation of data preprocessing logic from the dataset class using callable objects.

### 10. Softmax & Cross-Entropy Loss (`11_softmax_and_crossentropy.py`)
Derives softmax from scratch in NumPy, then contrasts with `torch.softmax`. Explains `nn.CrossEntropyLoss` as a fused `LogSoftmax + NLLLoss` — emphasizing that it expects **raw logits**, not softmax outputs. Covers both binary (`BCELoss + Sigmoid`) and multiclass (`CrossEntropyLoss`, no final activation) network designs.

### 11. Activation Functions (`12_activation_functions.py`, `12_plot_activations.py`)
Covers Sigmoid, Tanh, ReLU, Leaky ReLU, and Binary Step — implemented via both `nn.Module` subclasses and `torch.nn.functional` calls. Matplotlib plots illustrate the shape and saturation behavior of each function, motivating ReLU's preference in deep networks.

### 12. Feedforward Neural Network (`13_feedforward.py`)
Trains a two-layer fully connected network on **MNIST** (784 → 500 → 10). Demonstrates flattening image tensors with `.reshape(-1, 28*28)`, `nn.CrossEntropyLoss`, the Adam optimizer, and device-agnostic code with `torch.device('cuda' if ... else 'cpu')`.

### 13. Convolutional Neural Network (`14_cnn.py`)
Implements `ConvNet` on **CIFAR-10** with two `Conv2d` + `MaxPool2d` blocks followed by three fully connected layers. Tracks per-class accuracy across 10 categories. Saves the trained model with `torch.save(model.state_dict(), PATH)`.

### 14. Transfer Learning (`15_transfer_learning.py`)
Two strategies on **ResNet-18** (ImageNet pretrained):
- **Fine-tuning**: all weights updated, final FC layer replaced for 2-class output.
- **Fixed feature extractor**: backbone frozen via `requires_grad = False`, only the new FC head trained.

Uses `lr_scheduler.StepLR` for learning rate decay and `copy.deepcopy` to retain best validation weights.

### 15. TensorBoard Integration (`16_tensorboard.py`)
Integrates `SummaryWriter` to log training loss, accuracy, model graph (`writer.add_graph`), image grids, and per-class **precision-recall curves** (`writer.add_pr_curve`) for the 10 MNIST classes. Demonstrates structured experiment tracking with named run directories.

### 16. Model Saving & Loading (`17_save_load.py`)
Covers three persistence patterns:
- **Full model save** (`torch.save(model, PATH)`) — convenient but fragile.
- **State dict save** (`torch.save(model.state_dict(), PATH)`) — recommended, portable.
- **Checkpoint save** — stores epoch, model state, and optimizer state for resumable training.

Includes guidance on CPU ↔ GPU serialization using `map_location`.

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/syedrafayme143/pytorch-deep-learning-fundamentals.git
cd pytorch-deep-learning-fundamentals

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r 01_requirements.txt
```

---

## 🚀 Running the Scripts

```bash
# Example: run the feedforward MNIST network
python 13_feedforward.py

# Example: run the CNN on CIFAR-10
python 14_cnn.py

# Launch TensorBoard (after running 16_tensorboard.py)
tensorboard --logdir=runs
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Runtime |
| PyTorch | 2.x | Deep learning framework |
| torchvision | 0.15+ | Datasets, transforms, pretrained models |
| NumPy | 1.24+ | Array operations & data handling |
| scikit-learn | 1.2+ | Datasets (regression, breast cancer) |
| Matplotlib | 3.7+ | Visualization |
| TensorBoard | 2.x | Training metrics & experiment tracking |

---

## 📚 Learning Outcomes

By working through this repository, the following competencies are demonstrated:

- Building and training models end-to-end using the PyTorch `nn.Module` API
- Understanding **automatic differentiation** via computation graphs
- Designing custom `Dataset` + `DataLoader` pipelines for scalable training
- Implementing **CNNs** and applying **transfer learning** with pretrained models
- Monitoring training with **TensorBoard** (scalars, images, PR curves)
- Applying best practices for **model serialization** and checkpoint-based training resumption

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
