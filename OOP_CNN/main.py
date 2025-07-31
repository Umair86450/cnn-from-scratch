# main.py
from simple_cnn import SimpleCNN

def main():
    image_path = "tree.jpeg"  # Change path as needed
    label = 1                # Example label (you can adjust)
    epochs = 5
    learning_rate = 0.1

    # Create CNN model instance
    cnn = SimpleCNN(input_path=image_path, label=label)

    # Train the CNN
    cnn.train(epochs=epochs, learning_rate=learning_rate)

if __name__ == "__main__":
    main()
