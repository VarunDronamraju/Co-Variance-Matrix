import numpy as np

class NeuralNetwork:
    def _init_(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.biases_input_hidden = np.random.randn(1, self.hidden_size)
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.biases_hidden_output = np.random.randn(1, self.output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        # Input to hidden layer
        self.hidden_sum = np.dot(inputs, self.weights_input_hidden) + self.biases_input_hidden
        self.hidden_activation = self.sigmoid(self.hidden_sum)
        
        # Hidden to output layer
        self.output_sum = np.dot(self.hidden_activation, self.weights_hidden_output) + self.biases_hidden_output
        self.output_activation = self.sigmoid(self.output_sum)
        
        return self.output_activation
    
    def backward(self, inputs, targets, learning_rate):
        # Compute output layer error
        output_error = targets - self.output_activation
        output_delta = output_error * self.sigmoid_derivative(self.output_activation)
        
        # Compute hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_activation)
        
        # Update weights and biases
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_activation.T, output_delta)
        self.biases_hidden_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        self.biases_input_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

# Example usage:
# Define input, hidden, and output sizes
input_size = 2
hidden_size = 3
output_size = 1

# Create a neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Define input data and target output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the neural network
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    output = nn.forward(X)
    nn.backward(X, y, learning_rate)

# Test the trained neural network
print("Final Output after Training:")
print(nn.forward(X))
