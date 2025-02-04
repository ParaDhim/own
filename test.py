# S-Box (fixed mapping based on the table)
S_BOX = {
    0x0: 0xE, 0x1: 0x4, 0x2: 0xD, 0x3: 0x1,
    0x4: 0x2, 0x5: 0xF, 0x6: 0xB, 0x7: 0x8,
    0x8: 0x3, 0x9: 0xA, 0xA: 0x6, 0xB: 0xC,
    0xC: 0x5, 0xD: 0x9, 0xE: 0x0, 0xF: 0x7
}

#P-box permutation (transposition of bits)
P_BOX = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]

# Example Subkeys
SUBKEYS = [0x1F2A, 0x3C4D, 0x5E6F]

def s_box_substituion(nibble):
    """ Substitutes a 4-bit nibbke using the S-Box """
    return S_BOX[nibble]

def permutaion(block):
    """ Permutes a 16-bit block using the P-box """
    return sum(((block >> (i)) & 1) << (P_BOX[i] - 1) for i in range(16))

def key_mixing(block, subkey):
    """ Mixes a 16-bit block with a 16-bit subkey """
    return block ^ subkey

def round_function(block, subkey):
    """ Performs the operation of a single round: Substitution, Permutation, and Key Mixing """
    # Split block into 4 nibbles
    nibbles = [(block >> (i * 4)) & 0xF for i in range(4)]
    
    # Apply substitution to each nibble
    substituted = [s_box_substituion(nibble) for nibble in nibbles]
    
    # Concatenate the nibbles into a 16-bit block
    substituted_block = sum(substituted[i] << (i * 4) for i in range(4))
    
    # Permute the 16-bit block
    block = permutaion(substituted_block)
    
    # Mix the block with the subkey
    mix_block = key_mixing(block, subkey)
    
    return mix_block

def encrypt_block(plaintext):
    """ Encrypts a single block of plaintext using SPN cipher """    
    block = plaintext
    for subkey in SUBKEYS:
        block = round_function(block, subkey)
    return block

if __name__ == "__main__":
    plaintext = 0x1234
    ciphertext = encrypt_block(plaintext)
    print(f"Plaintext: {hex(plaintext)}")
    print(f"Ciphertext: {hex(ciphertext)}")



# Inverse S-box (inverse mapping of provided S-box)
INVERSE_S_BOX = {v: k for k, v in S_BOX.items()}

def inv_s_box_substitution(nibble):
    """Substitutes a 4-bit nibble using the inverse S-Box"""
    return INVERSE_S_BOX[nibble]

def inverse_permutation(block):
    """Permutes a 16-bit block using the inverse P-box"""
    return sum(((block >> (P_BOX.index(i))) & 1) << (i - 1) for i in range(1, 17))

def inv_round_function(block, subkey):
    """Performs the inverse operation of a single round"""
    # First, undo the key mixing (XOR is its own inverse)
    block = block ^ subkey
    
    # Undo the permutation
    block = inverse_permutation(block)
    
    # Split block into 4 nibbles
    nibbles = [(block >> (i * 4)) & 0xF for i in range(4)]
    
    # Apply inverse substitution to each nibble
    inverted = [inv_s_box_substitution(nibble) for nibble in nibbles]
    
    # Concatenate the nibbles into a 16-bit block
    block = sum(inverted[i] << (i * 4) for i in range(4))
    
    return block

def decrypt(ciphertext):
    """Decrypts a single block of ciphertext using SPN cipher"""
    block = ciphertext
    
    # Process in reverse order of subkeys
    for subkey in reversed(SUBKEYS):
        block = inv_round_function(block, subkey)
    
    return block



if __name__ == "__main__":
    plaintext = 0x1234
    ciphertext = encrypt_block(plaintext)
    decrypted_text = decrypt(ciphertext)
    print(f"Plaintext: {hex(plaintext)}")
    print(f"Ciphertext: {hex(ciphertext)}")
    print(f"Decrypted Text: {hex(decrypted_text)}")




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class CipherNet(nn.Module):
    def __init__(self):
        super(CipherNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class CipherDataset(Dataset):
    def __init__(self, input_pairs: List[Tuple[int, int]], labels: List[int]):
        self.input_pairs = torch.tensor([
            self._convert_pair_to_bits(c, c_prime) 
            for c, c_prime in input_pairs
        ], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def _convert_pair_to_bits(self, c: int, c_prime: int) -> List[int]:
        c_bits = [(c >> i) & 1 for i in range(16)]
        c_prime_bits = [(c_prime >> i) & 1 for i in range(16)]
        return c_bits + c_prime_bits

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_pairs[idx], self.labels[idx]

class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_accuracies = []
        
    def plot_training_progress(self):
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def plot_roc_curve(self, y_true, y_pred_prob):
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

def train_network(training_input: List[Tuple[int, int]], 
                 training_output: List[int], 
                 epochs: int,
                 val_split: float = 0.2) -> Tuple[CipherNet, TrainingVisualizer]:
    """
    Train the neural network on cipher data with visualization
    """
    visualizer = TrainingVisualizer()
    
    # Split data into train and validation sets
    n_samples = len(training_input)
    n_val = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)
    
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    train_input = [training_input[i] for i in train_idx]
    train_output = [training_output[i] for i in train_idx]
    val_input = [training_input[i] for i in val_idx]
    val_output = [training_output[i] for i in val_idx]
    
    # Create datasets and dataloaders
    train_dataset = CipherDataset(train_input, train_output)
    val_dataset = CipherDataset(val_input, val_output)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize network and move to GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = CipherNet().to(device)
    
    optimizer = optim.Adam(net.parameters())
    criterion = nn.BCELoss()

    # Training loop with progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        net.train()
        running_loss = 0.0
        batch_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")
        
        for inputs, labels in batch_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate validation accuracy
        net.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_true = []
        val_pred_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                val_pred_probs.extend(outputs.squeeze().cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        val_accuracy = val_correct / val_total
        
        visualizer.train_losses.append(epoch_loss)
        visualizer.val_accuracies.append(val_accuracy)
        
        epoch_pbar.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'val_acc': f'{val_accuracy:.4f}'
        })

    # Plot final training metrics
    visualizer.plot_training_progress()
    visualizer.plot_confusion_matrix(val_true, val_preds)
    visualizer.plot_roc_curve(val_true, val_pred_probs)
    
    return net, visualizer

def train(n: int, epochs: int, delta_x: int) -> Tuple[CipherNet, TrainingVisualizer]:
    """
    Implementation of the Train function from pseudocode with visualization
    """
    print("Generating training data...")
    training_input = []
    training_output = []
    
    for _ in tqdm(range(n), desc="Generating samples"):
        if np.random.random() > 0.5:
            m = np.random.randint(0, 2**16)
            k = np.random.randint(0, 2**16)
            
            c = encrypt_block(m)
            c_prime = encrypt_block(m ^ delta_x)
            
            training_input.append((c, c_prime))
            training_output.append(1)
        else:
            c = np.random.randint(0, 2**16)
            c_prime = np.random.randint(0, 2**16)
            
            training_input.append((c, c_prime))
            training_output.append(0)
    
    return train_network(training_input, training_output, epochs)

def neural_distinguisher(delta_x: int, n: int, tau_n: int, epochs: int) -> bool:
    """
    Implementation of the Neural distinguisher A3 from pseudocode with added accuracy metrics
    """
    print("Training neural distinguisher...")
    net, visualizer = train(n, epochs, delta_x)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = net.to(device)
    
    print("Testing neural distinguisher...")
    h = 0
    test_results = []
    true_labels = []
    
    # Test with balanced positive and negative samples
    for i in tqdm(range(n), desc="Testing"):
        m = np.random.randint(0, 2**16)
        
        if i < n/2:  # Positive samples (related pairs)
            c = encrypt_block(m)
            c_prime = encrypt_block(m ^ delta_x)
            true_labels.append(1)
        else:  # Negative samples (random pairs)
            c = np.random.randint(0, 2**16)
            c_prime = np.random.randint(0, 2**16)
            true_labels.append(0)
        
        test_input = CipherDataset([(c, c_prime)], [0]).input_pairs
        test_input = test_input.to(device)
        
        with torch.no_grad():
            prediction = net(test_input)
            if prediction.item() > 0.5:
                h += 1
            test_results.append(prediction.item())
    
    # Calculate accuracy metrics
    predictions = [1 if x > 0.5 else 0 for x in test_results]
    correct = sum(1 for x, y in zip(predictions, true_labels) if x == y)
    accuracy = correct / len(predictions)
    
    # Calculate separate accuracies for positive and negative cases
    positive_correct = sum(1 for x, y in zip(predictions[:n//2], true_labels[:n//2]) if x == y)
    negative_correct = sum(1 for x, y in zip(predictions[n//2:], true_labels[n//2:]) if x == y)
    positive_accuracy = positive_correct / (n//2)
    negative_accuracy = negative_correct / (n//2)
    
    print(f"\nAccuracy Metrics:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Related Pairs Accuracy: {positive_accuracy:.4f}")
    print(f"Random Pairs Accuracy: {negative_accuracy:.4f}")
    
    # Plot distribution of test predictions
    plt.figure(figsize=(8, 6))
    plt.hist([x for i, x in enumerate(test_results) if true_labels[i] == 1], 
             bins=25, alpha=0.5, label='Related Pairs', density=True)
    plt.hist([x for i, x in enumerate(test_results) if true_labels[i] == 0], 
             bins=25, alpha=0.5, label='Random Pairs', density=True)
    plt.title('Distribution of Neural Distinguisher Predictions')
    plt.xlabel('Prediction Value')
    plt.ylabel('Density')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.legend()
    plt.show()
    
    return h >= tau_n

if __name__ == "__main__":
    # Parameters
    DELTA_X = 0x0040
    N = 500000  # Increased sample size
    TAU_N = 850  # Adjusted threshold
    EPOCHS = 150  # More training epochs
    
    # Run neural distinguisher with visualization
    result = neural_distinguisher(DELTA_X, N, TAU_N, EPOCHS)
    print(f"Neural distinguisher result: {result}")