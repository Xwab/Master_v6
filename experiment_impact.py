import numpy as np
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size, init_seed):
        self.rng = np.random.RandomState(init_seed)
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W1 = self.rng.uniform(-limit, limit, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        limit = np.sqrt(6 / (hidden_size + output_size))
        self.W2 = self.rng.uniform(-limit, limit, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = softmax(self.z2)
        return self.output

    def backward(self, X, y, learning_rate):
        batch_size = X.shape[0]
        
        # Output layer error (Cross Entropy with Softmax derivative simplification)
        # dL/dz2 = output - y_one_hot
        d_z2 = self.output - y
        
        d_W2 = np.dot(self.a1.T, d_z2) / batch_size
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) / batch_size
        
        # Hidden layer error
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * sigmoid_derivative(self.a1)
        
        d_W1 = np.dot(X.T, d_z1) / batch_size
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) / batch_size
        
        # Update weights
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        
        return np.mean(-np.sum(y * np.log(self.output + 1e-8), axis=1))

def generate_data(n_samples=1000, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 20) # 20 features
    # Create a non-linear relationship with noise
    y_logits = np.sum(X[:, :5]**2, axis=1) - np.sum(X[:, 5:10]**2, axis=1) + rng.randn(n_samples) * 2.0
    y = (y_logits > 0).astype(int)
    # Convert to one-hot
    y_one_hot = np.zeros((n_samples, 2))
    y_one_hot[np.arange(n_samples), y] = 1
    return X, y_one_hot

def train_experiment(init_seed, data_order_seed, epochs=10, batch_size=32):
    # Fixed data content, variable order
    X, y = generate_data(n_samples=1000, seed=42)
    
    # Shuffle data based on data_order_seed
    data_rng = np.random.RandomState(data_order_seed)
    indices = np.arange(X.shape[0])
    
    model = SimpleMLP(input_size=20, hidden_size=64, output_size=2, init_seed=init_seed)
    
    final_loss = 0
    lr = 0.1
    
    for epoch in range(epochs):
        # Shuffle for this training run (determined by data_order_seed logic)
        # Note: Usually we shuffle every epoch. 
        # Here we fix the *sequence of batches* for the entire training based on the seed
        # To make "Data Loading Order" deterministic per seed, we re-seed the shuffler or shuffle once.
        # Let's shuffle once per experiment to represent "Different Data Order".
        # If we shuffle every epoch with a fixed seed generator, it's a specific trajectory.
        
        # To strictly test "Data Loading Order", we use the data_rng to shuffle indices differently
        # at the start, or use it to seed the per-epoch shuffle.
        # Let's simple shuffle once at start for simplicity, or shuffle every epoch using the rng.
        data_rng.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            model.forward(X_batch)
            
            # Backward pass
            loss = model.backward(X_batch, y_batch, lr)
            final_loss = loss
            
    # Return final loss on training set (convergence metric)
    # And validation loss on a fixed set
    X_val, y_val = generate_data(n_samples=200, seed=999)
    val_preds = model.forward(X_val)
    val_loss = np.mean(-np.sum(y_val * np.log(val_preds + 1e-8), axis=1))
    
    # Calculate accuracy
    preds = np.argmax(val_preds, axis=1)
    targets = np.argmax(y_val, axis=1)
    accuracy = np.mean(preds == targets)
    
    return val_loss, accuracy

def main():
    print("Running experiment to compare impact of Initialization vs Data Loading Order...")
    print("Task: Binary classification on synthetic data (MLP 20->64->2)")
    print("-" * 60)
    
    base_init_seed = 100
    base_data_seed = 200
    n_runs = 10
    
    # 1. Vary Initialization (Fix Data Order)
    print(f"1. Varying Initialization (Fixed Data Seed={base_data_seed})")
    init_results_acc = []
    init_results_loss = []
    for i in range(n_runs):
        seed = base_init_seed + i
        loss, acc = train_experiment(init_seed=seed, data_order_seed=base_data_seed)
        init_results_acc.append(acc)
        init_results_loss.append(loss)
        # print(f"   Run {i}: Init Seed {seed} -> Val Acc: {acc:.4f}, Loss: {loss:.4f}")
    
    init_std_acc = np.std(init_results_acc)
    init_std_loss = np.std(init_results_loss)
    
    # 2. Vary Data Order (Fix Initialization)
    print(f"2. Varying Data Order (Fixed Init Seed={base_init_seed})")
    data_results_acc = []
    data_results_loss = []
    for i in range(n_runs):
        seed = base_data_seed + i
        loss, acc = train_experiment(init_seed=base_init_seed, data_order_seed=seed)
        data_results_acc.append(acc)
        data_results_loss.append(loss)
        # print(f"   Run {i}: Data Seed {seed} -> Val Acc: {acc:.4f}, Loss: {loss:.4f}")
        
    data_std_acc = np.std(data_results_acc)
    data_std_loss = np.std(data_results_loss)
    
    print("-" * 60)
    print("RESULTS Summary (Standard Deviation across 10 runs):")
    print(f"Varying Initialization - Acc Std: {init_std_acc:.6f}, Loss Std: {init_std_loss:.6f}")
    print(f"Varying Data Order     - Acc Std: {data_std_acc:.6f}, Loss Std: {data_std_loss:.6f}")
    print("-" * 60)
    
    if init_std_acc > data_std_acc:
        print(">> CONCLUSION: Model Initialization had a GREATER impact on result variance.")
    else:
        print(">> CONCLUSION: Data Loading Order had a GREATER impact on result variance.")

if __name__ == "__main__":
    main()
