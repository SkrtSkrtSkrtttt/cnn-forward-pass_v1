# cnn-forward-pass_v1
A C++ implementation of a simplified Convolutional Neural Network (CNN) that performs forward propagation across multiple layers to produce classification probabilities for 10 object classes. This project mirrors the architecture of ConvNetJS (CIFAR-10) by Andrej Karpathy, 



---
🧩 Core Architecture

**Goal:**  
Simulate forward propagation of a convolutional neural network without training or backpropagation, using predefined weights or synthetic data.

**CNN Structure:**

1. **Input Layer**  
   - 32x32x3 RGB image

2. **Conv Layer 1 + ReLU + Max Pool**  
   - Conv: 5x5x3 filters × 16 channels → 32x32x16  
   - ReLU: max(0, x)  
   - Pool: 2x2, stride 2 → 16x16x16

3. **Conv Layer 2 + ReLU + Max Pool**  
   - Conv: 5x5x16 filters × 20 channels → 16x16x20  
   - Pool: → 8x8x20

4. **Conv Layer 3 + ReLU + Max Pool**  
   - Conv: 5x5x20 filters × 20 channels → 8x8x20  
   - Pool: → 4x4x20

5. **Fully Connected + Softmax**  
   - Flattened input: 4x4x20 → Dense: 10  
   - Output: Softmax vector (1x10)

---

## 🧠 Components

Each layer is modularized and implemented using standard C++ constructs:

- **Convolution Layer**: Applies 4D filters using zero-padding and stride.
- **ReLU Layer**: Applies activation function: `max(0, x)`.
- **Max Pooling**: Reduces spatial dimensions via 2D pooling on each channel.
- **Fully Connected Layer**: Computes dot products between flattened feature maps and weight matrices.
- **Softmax Layer**: Normalizes logits into a probability distribution.

---

## 📂 Data Format

Synthetic or externally sourced data is organized as:

- `D1`: Input image → 32x32x3
- `D3`, `D8`, `D13`: Conv filters
- `D4`, `D9`, `D14`: Bias vectors
- `D18`: Fully connected weights (4x4x20x10)
- `D19`: FC layer bias (1x10)


---

## Notes
-Needs additonal testing
Last commit: 8/1/25

---
## Author
Naafiul Hossain
