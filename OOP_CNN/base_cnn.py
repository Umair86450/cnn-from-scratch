from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

class BaseCNN(ABC):
    def __init__(self, input_path, label, num_classes=2):
        self._image = self._load_image(input_path)
        self._label = label
        self._num_classes = num_classes
        self._kernel = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ], dtype=float)
        self._fc_weights = None
        self._fc_biases = None

    def _load_image(self, path):
        image = Image.open(path).convert("L").resize((28, 28))
        image_np = np.array(image, dtype=float) / 255.0
        print(f"Input Image shape: {image_np.shape}")
        print(f"Input Image sample data (5x5):\n{image_np[:5, :5]}")
        return image_np

    @staticmethod
    def relu(x):
        activated = np.maximum(0, x)
        print(f"\nAfter ReLU (shape {activated.shape}): sample 5x5 values:\n{activated[:5, :5]}")
        return activated

    @staticmethod
    def max_pooling(x, size=2, stride=2):
        h, w = x.shape
        pooled = np.zeros((h // 2, w // 2))
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                pooled[i // 2, j // 2] = np.max(x[i:i + size, j:j + size])
        print(f"\nAfter Max Pooling (shape {pooled.shape}): sample 5x5 values:\n{pooled[:5, :5]}")
        return pooled

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        sm = e_x / np.sum(e_x)
        print(f"\nSoftmax probabilities: {sm}")
        return sm

    @staticmethod
    def cross_entropy_loss(probs, target):
        loss = -np.log(probs[target] + 1e-8)
        print(f"Cross-Entropy Loss: {loss}")
        return loss

    @staticmethod
    def cross_entropy_derivative(probs, target):
        grad = probs.copy()
        grad[target] -= 1
        return grad

    @abstractmethod
    def convolve2d(self, img, kernel):
        pass

    @abstractmethod
    def train(self, epochs=5, learning_rate=0.1):
        pass
