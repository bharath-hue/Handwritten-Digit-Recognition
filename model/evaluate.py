import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix as a heatmap.
    """
    figure = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Add labels to the plot
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations to each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot
    os.makedirs('model/plots', exist_ok=True)
    plt.savefig('model/plots/confusion_matrix.png')
    plt.close()

def plot_misclassified_examples(test_images, test_labels, predictions, n=10):
    """
    Plot some misclassified examples.
    """
    # Find misclassified examples
    misclassified_indices = np.where(predictions != test_labels)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return
    
    # Select a random subset of misclassified examples
    indices_to_plot = np.random.choice(misclassified_indices, 
                                      min(n, len(misclassified_indices)), 
                                      replace=False)
    
    # Plot the examples
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices_to_plot):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {test_labels[idx]}, Pred: {predictions[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('model/plots/misclassified_examples.png')
    plt.close()

def main():
    # Load the test dataset
    print("Loading test dataset...")
    (_, _), (test_images, test_labels) = mnist.load_data()
    
    # Normalize and reshape the test images
    test_images = test_images.astype('float32') / 255
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    
    # Load the trained model
    print("Loading trained model...")
    model = load_model('model/saved_model/mnist_cnn.h5')
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Make predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_labels
    
    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, 
                               target_names=[str(i) for i in range(10)]))
    
    # Plot misclassified examples
    plot_misclassified_examples(test_images, true_classes, predicted_classes)
    
    print("Evaluation complete. Results saved to model/plots/")

if __name__ == "__main__":
    main()