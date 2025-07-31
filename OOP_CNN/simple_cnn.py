import numpy as np
from base_cnn import BaseCNN

class SimpleCNN(BaseCNN):
    def __init__(self, input_path, label, num_classes=2):
        super().__init__(input_path, label, num_classes)
        np.random.seed(42)
        self._fc_weights = np.random.randn(num_classes, 169) * 0.01  # assuming flattened size is 13x13
        self._fc_biases = np.zeros(num_classes)

    def convolve2d(self, img, kernel):
        k = kernel.shape[0]
        out_dim = img.shape[0] - k + 1
        out = np.zeros((out_dim, out_dim))
        for i in range(out_dim):
            for j in range(out_dim):
                region = img[i:i + k, j:j + k]
                out[i, j] = np.sum(region * kernel)
        print(f"\nAfter Convolution (shape {out.shape}): sample 5x5 values:\n{out[:5, :5]}")
        return out

    def train(self, epochs=5, learning_rate=0.1):
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch} =====")
            conv_out = self.convolve2d(self._image, self._kernel)
            activated = self.relu(conv_out)
            pooled = self.max_pooling(activated)
            flattened = pooled.flatten()

            logits = np.dot(self._fc_weights, flattened) + self._fc_biases
            print(f"Logits before softmax: {logits}")
            probs = self.softmax(logits)
            loss = self.cross_entropy_loss(probs, self._label)

            # Backpropagation
            grad_logits = self.cross_entropy_derivative(probs, self._label)
            grad_weights = grad_logits[:, None] * flattened[None, :]
            grad_biases = grad_logits

            self._fc_weights -= learning_rate * grad_weights
            self._fc_biases -= learning_rate * grad_biases

            pred_class = np.argmax(probs)
            print(f"Predicted class: {pred_class} | Correct: {pred_class == self._label}")
