import numpy as np
from torchvision.datasets import MNIST
from sklearn.preprocessing import OneHotEncoder


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image / 255.0)  # Normalize the image data - mnist max pixel value is 255
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

input_size = 784
hidden_size = 100
output_size = 10
learning_rate = 0.01

one_hot_encoder = OneHotEncoder(sparse_output=False)
train_Y_one_hot = one_hot_encoder.fit_transform(np.array(train_Y).reshape(-1, 1))
test_Y_one_hot = one_hot_encoder.transform(np.array(test_Y).reshape(-1, 1))

# Xavier initialization following fan_in and fan_out from the slide
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / (input_size + hidden_size))
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
b2 = np.zeros((1, output_size))

# we will use tanh on the hidden layer and softmax on the output layer
def tanh_activation(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - (np.tanh(x) ** 2)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2):
    # Layer 1 (hidden layer)
    Z1 = X.dot(W1) + b1
    Y1 = tanh_activation(Z1)

    # Layer 2 (output layer)
    Z2 = Y1.dot(W2) + b2
    Y2 = softmax(Z2)

    return Y1, Y2

regularization_lambda = 0.001  # Regularization parameter for L2 regularization

def backward_propagation(X, y, Y1, Y2, W2, learning_rate):
    global W1, b1, b2

    # Output layer gradients
    m = y.shape[0]
    gradient_output = - (y - Y2) # gradient of loss function - cross entropy loss
    dW2 = (Y1.T.dot(gradient_output) + regularization_lambda * W2) / m # average gradient of the loss w.r.t. W2 appliyng L2 regularization
    db2 = np.sum(gradient_output, axis=0, keepdims=True) / m

    # Hidden layer gradients
    dY1 = gradient_output.dot(W2.T) 
    gradient_hidden = dY1 * tanh_derivative(Y1) # propagate the gradient w.r.t. the next layer
    dW1 = (X.T.dot(gradient_hidden) + regularization_lambda * W1) / m
    db1 = np.sum(gradient_hidden, axis=0, keepdims=True) / m

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

def train_NN(X_train, y_train, epochs, batch_size):
    global W1, b1, W2, b2

    for epoch in range(epochs):
        # Shuffle
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            Y1, Y2 = forward_propagation(X_batch, W1, b1, W2, b2)
            W1, b1, W2, b2 = backward_propagation(X_batch, y_batch, Y1, Y2, W2, learning_rate)
        
        val_accuracy = evaluate(X_train, y_train)
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy * 100:.2f}%")

def evaluate(X, y):
    _, Y2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(Y2, axis=1)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

train_NN(train_X, train_Y_one_hot, epochs=500, batch_size=100)
validation_accuracy = evaluate(test_X, test_Y_one_hot)
print(f"Validation Accuracy for the test set: {validation_accuracy * 100:.2f}%")