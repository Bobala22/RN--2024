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

one_hot_encoder = OneHotEncoder(sparse_output=False)
train_Y_one_hot = one_hot_encoder.fit_transform(np.array(train_Y).reshape(-1, 1))
test_Y_one_hot = one_hot_encoder.transform(np.array(test_Y).reshape(-1, 1))

# Initialize weights and biases
W = np.random.randn(784, 10) * 0.01  # 28 x 28 = 784 features
b = np.zeros(10)

# Define softmax function
def softmax(z):
    max_z = np.max(z, axis=1).reshape(-1, 1) # axis = 1, compute on the column, where the classes are
    exp_z = np.exp(z - max_z)

    sum_exp_z = np.sum(exp_z, axis=1).reshape(-1, 1)

    return exp_z / sum_exp_z


# Define forward propagation
def forward_propagation(X, W, b):
    z = np.dot(X, W) + b 
    y_hat = softmax(z)
    return y_hat

# Define cross entropy loss
def cross_entropy_loss(y_hat, y):
    m = y.shape[0]
    log_pred = -np.log(y_hat[range(m), np.argmax(y, axis=1)])
    return np.sum(log_pred) / m

# Define backward propagation
def backward_propagation(X, y_hat, y, W, b, learning_rate):
    m = y.shape[0]
    gradient = y_hat - y
    dW = np.dot(X.T, gradient) / m  # Gradient with respect to weights
    db = np.sum(gradient, axis=0) / m  # Gradient with respect to biases
    
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

# Define training function
def train_perceptron(train_X, train_Y, W, b, epochs=100, batch_size=100, learning_rate=0.1):
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(train_X.shape[0])
        train_X_shuffled = train_X[shuffled_indices]
        train_Y_shuffled = train_Y[shuffled_indices]

        for i in range(0, train_X.shape[0], batch_size):
            X_batch = train_X_shuffled[i:i + batch_size]
            Y_batch = train_Y_shuffled[i:i + batch_size]
            
            y_hat = forward_propagation(X_batch, W, b)
            
            W, b = backward_propagation(X_batch, y_hat, Y_batch, W, b, learning_rate)

        # Eval
        y_hat_train = forward_propagation(train_X, W, b)
        loss = cross_entropy_loss(y_hat_train, train_Y)
        accuracy = np.mean(np.argmax(y_hat_train, axis=1) == np.argmax(train_Y, axis=1))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    
    return W, b

def test_perceptron(test_X, test_Y, W, b):
    y_hat_test = forward_propagation(test_X, W, b)
    accuracy = np.mean(np.argmax(y_hat_test, axis=1) == np.argmax(test_Y, axis=1))
    print(f"Test Accuracy: {accuracy*100:.2f}%")

W, b = train_perceptron(train_X, train_Y_one_hot, W, b, epochs=100, batch_size=100, learning_rate=0.01)

test_perceptron(test_X, test_Y_one_hot, W, b)