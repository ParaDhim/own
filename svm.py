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






import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import random

# Using the provided cipher components
S_BOX = {
    0x0: 0xE, 0x1: 0x4, 0x2: 0xD, 0x3: 0x1,
    0x4: 0x2, 0x5: 0xF, 0x6: 0xB, 0x7: 0x8,
    0x8: 0x3, 0x9: 0xA, 0xA: 0x6, 0xB: 0xC,
    0xC: 0x5, 0xD: 0x9, 0xE: 0x0, 0xF: 0x7
}

P_BOX = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]

def calculate_differential_profile():
    """Calculate differential characteristics of the S-box"""
    diff_profile = np.zeros((16, 16), dtype=int)
    
    for input_val in range(16):
        for input_diff in range(16):
            output_val = S_BOX[input_val]
            output_val_2 = S_BOX[input_val ^ input_diff]
            output_diff = output_val ^ output_val_2
            diff_profile[input_diff][output_diff] += 1
            
    return diff_profile

def generate_differential_pairs(num_pairs, input_diff):
    """Generate plaintext pairs with specific input difference"""
    pairs = []
    for _ in range(num_pairs):
        x1 = random.randint(0, 0xFFFF)
        x2 = x1 ^ input_diff
        pairs.append((x1, x2))
    return pairs

def extract_features(ciphertext1, ciphertext2):
    """Extract differential features from ciphertext pair"""
    features = []
    
    # Extract nibble differences
    for i in range(4):
        nibble1 = (ciphertext1 >> (i * 4)) & 0xF
        nibble2 = (ciphertext2 >> (i * 4)) & 0xF
        diff = nibble1 ^ nibble2
        features.append(diff)
    
    # Add position-based features
    for pos in range(16):
        bit1 = (ciphertext1 >> pos) & 1
        bit2 = (ciphertext2 >> pos) & 1
        features.append(bit1 ^ bit2)
    
    return features

def prepare_training_data(num_samples=1000):
    """Prepare training data for SVM with multiple key classes"""
    X = []  # Features
    y = []  # Labels (last round key nibbles)
    
    # Generate multiple different subkey sets
    test_subkeys = [
        [0x1F2A, 0x3C4D, 0x5E6F],
        [0x2B3A, 0x4D5E, 0x6F1A],
        [0x3C4D, 0x5E6F, 0x1A2B],
        [0x4D5E, 0x6F1A, 0x2B3C]
    ]
    
    input_differences = [0x1, 0x2, 0x4, 0x8]  # Common input differences
    
    def encrypt_with_keys(plaintext, keys):
        """Encrypt plaintext with given set of keys"""
        block = plaintext
        for key in keys:
            # Simulate the round function with these keys
            block = round_function(block, key)
        return block
    
    for _ in range(num_samples // len(test_subkeys)):
        plaintext = random.randint(0, 0xFFFF)
        
        for keys in test_subkeys:
            for input_diff in input_differences:
                features = []
                p1 = plaintext
                p2 = plaintext ^ input_diff
                
                # Encrypt both plaintexts with current key set
                c1 = encrypt_with_keys(p1, keys)
                c2 = encrypt_with_keys(p2, keys)
                
                # Extract features
                features.extend(extract_features(c1, c2))
                
                X.append(features)
                y.append(keys[-1] & 0xF)  # Last round key nibble
    
    return np.array(X), np.array(y)

def train_svm_classifier():
    """Train SVM classifier for key recovery"""
    print("Preparing training data...")
    X, y = prepare_training_data(2000)
    
    print(f"Number of classes in training data: {len(np.unique(y))}")
    print(f"Class distribution: {Counter(y)}")
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training SVM classifier...")
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_scaled, y)
    
    return svm, scaler

def analyze_ciphertext_pairs(pairs, svm, scaler):
    """Analyze ciphertext pairs using trained SVM"""
    X_test = []
    
    for p1, p2 in pairs:
        c1 = encrypt_block(p1)
        c2 = encrypt_block(p2)
        features = extract_features(c1, c2)
        X_test.append(features)
    
    X_test_scaled = scaler.transform(X_test)
    predictions = svm.predict(X_test_scaled)
    probabilities = svm.predict_proba(X_test_scaled)
    
    return predictions, probabilities

def visualize_results(predictions, probabilities):
    """Create visualizations of the analysis results"""
    # Plot 1: Key Predictions
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=16, alpha=0.7)
    plt.title('Distribution of Predicted Key Nibbles')
    plt.xlabel('Key Nibble Value')
    plt.ylabel('Frequency')
    
    # Plot 2: Prediction Probabilities
    plt.figure(figsize=(12, 6))
    plt.imshow(probabilities.T, aspect='auto', cmap='hot')
    plt.colorbar(label='Probability')
    plt.title('Key Nibble Prediction Probabilities')
    plt.xlabel('Test Sample')
    plt.ylabel('Key Value')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Starting differential cryptanalysis with SVM...")
    
    # Train SVM classifier
    svm, scaler = train_svm_classifier()
    
    # Generate test pairs
    input_diff = 0x1
    test_pairs = generate_differential_pairs(100, input_diff)
    
    # Analyze pairs
    predictions, probabilities = analyze_ciphertext_pairs(test_pairs, svm, scaler)
    
    # Calculate accuracy
    true_key_nibble = SUBKEYS[-1] & 0xF
    accuracy = accuracy_score([true_key_nibble] * len(predictions), predictions)
    
    print(f"\nAnalysis Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nPredicted key nibble distribution:")
    print(Counter(predictions))
    
    # Visualize results
    visualize_results(predictions, probabilities)







import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy.interpolate import interp1d
import random
from tqdm import tqdm


def enhanced_extract_features(ciphertext1, ciphertext2):
    """Enhanced feature extraction with additional differential patterns"""
    features = []
    
    # Extract nibble differences with position weights
    for i in range(4):
        nibble1 = (ciphertext1 >> (i * 4)) & 0xF
        nibble2 = (ciphertext2 >> (i * 4)) & 0xF
        diff = nibble1 ^ nibble2
        features.append(diff)
        
        # Add Hamming weight of the difference
        features.append(bin(diff).count('1'))
        
    # Add bit-level differences with position information
    for pos in range(16):
        bit1 = (ciphertext1 >> pos) & 1
        bit2 = (ciphertext2 >> pos) & 1
        features.append(bit1 ^ bit2)
        
    # Add adjacent bit pair differences
    for pos in range(15):
        pair1 = (ciphertext1 >> pos) & 0x3
        pair2 = (ciphertext2 >> pos) & 0x3
        features.append(pair1 ^ pair2)
    
    return features

def prepare_enhanced_training_data(num_samples=5000):
    """Prepare enhanced training data with better diversity"""
    X = []
    y = []
    
    # Extended set of test subkeys
    test_subkeys = [
        [0x1F2A, 0x3C4D, 0x5E6F]
    ]
    
    # Extended set of input differences
    input_differences = [0x1, 0x2, 0x4, 0x8, 0x3, 0x7, 0xF]
    
    def encrypt_with_keys(plaintext, keys):
        block = plaintext
        for key in keys:
            block = round_function(block, key)
        return block
    
    # Calculate total iterations for progress bar
    total_iterations = (num_samples // len(test_subkeys)) * len(test_subkeys) * len(input_differences)
    
    with tqdm(total=total_iterations, desc="Generating training data") as pbar:
        for _ in range(num_samples // len(test_subkeys)):
            for keys in test_subkeys:
                plaintext = random.randint(0, 0xFFFF)
                
                for input_diff in input_differences:
                    p1 = plaintext
                    p2 = plaintext ^ input_diff
                    
                    c1 = encrypt_with_keys(p1, keys)
                    c2 = encrypt_with_keys(p2, keys)
                    
                    features = enhanced_extract_features(c1, c2)
                    X.append(features)
                    y.append(keys[-1] & 0xF)
                    pbar.update(1)
    
    return np.array(X), np.array(y)

def train_enhanced_svm_classifier():
    """Train enhanced SVM classifier with better parameters"""
    print("Preparing enhanced training data...")
    X, y = prepare_enhanced_training_data(5000)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training enhanced SVM classifier...")
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        random_state=42,
        verbose=True  # Enable verbose output for SVM training
    )
    
    with tqdm(total=100, desc="Training SVM", bar_format='{l_bar}{bar}| {n:.0f}%') as pbar:
        # Fit the model and update progress periodically
        class ProgressCallback:
            def __init__(self, pbar):
                self.pbar = pbar
                self.last_progress = 0
                
            def __call__(self, progress):
                delta = progress - self.last_progress
                if delta > 0:
                    self.pbar.update(delta * 100)
                    self.last_progress = progress
        
        svm.fit(X_train_scaled, y_train)
        pbar.update(100 - pbar.n)  # Ensure we reach 100%
    
    return svm, scaler, X_test_scaled, y_test, X_train_scaled, y_train

def analyze_ciphertext_pairs(pairs, svm, scaler):
    """Analyze ciphertext pairs using trained SVM"""
    X_test = []
    
    for p1, p2 in tqdm(pairs, desc="Analyzing ciphertext pairs"):
        c1 = encrypt_block(p1)
        c2 = encrypt_block(p2)
        features = enhanced_extract_features(c1, c2)
        X_test.append(features)
    
    X_test_scaled = scaler.transform(X_test)
    predictions = svm.predict(X_test_scaled)
    probabilities = svm.predict_proba(X_test_scaled)
    
    return predictions, probabilities

def plot_roc_curves(svm, X_test_scaled, y_test):
    """Plot ROC curves for multi-class classification"""
    n_classes = len(np.unique(y_test))
    
    print("Calculating ROC curves...")
    # Binarize the labels for ROC curve calculation
    y_test_bin = label_binarize(y_test, classes=range(16))
    
    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_score = svm.predict_proba(X_test_scaled)
    
    for i in tqdm(range(n_classes), desc="Computing ROC curves"):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    
    for i, color in zip(range(n_classes), colors):
        if i in [0xF, 0xA, 0xB, 0xC]:  # Plot only for the key classes we're interested in
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve (class {i:X}, AUC = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves for Key Nibble Prediction')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def visualize_results(predictions, probabilities):
    """Create visualizations of the analysis results"""
    print("Generating visualizations...")
    # Plot 1: Key Predictions
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=16, alpha=0.7)
    plt.title('Distribution of Predicted Key Nibbles')
    plt.xlabel('Key Nibble Value')
    plt.ylabel('Frequency')
    
    # Plot 2: Prediction Probabilities
    plt.figure(figsize=(12, 6))
    plt.imshow(probabilities.T, aspect='auto', cmap='hot')
    plt.colorbar(label='Probability')
    plt.title('Key Nibble Prediction Probabilities')
    plt.xlabel('Test Sample')
    plt.ylabel('Key Value')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Starting enhanced differential cryptanalysis with SVM...")
    
    # Train enhanced SVM classifier
    svm, scaler, X_test_scaled, y_test, X_train_scaled, y_train = train_enhanced_svm_classifier()
    
    # Calculate and print accuracies
    print("\nCalculating accuracies...")
    train_accuracy = accuracy_score(y_train, svm.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, svm.predict(X_test_scaled))
    
    print(f"\nAnalysis Results:")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    
    # Generate test pairs and analyze
    input_diff = 0x1
    test_pairs = generate_differential_pairs(100, input_diff)
    predictions, probabilities = analyze_ciphertext_pairs(test_pairs, svm, scaler)
    
    print("\nPredicted key nibble distribution:")
    print(Counter(predictions))
    
    # Plot ROC curves
    plot_roc_curves(svm, X_test_scaled, y_test)
    
    # Visualize results
    visualize_results(predictions, probabilities)

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy.interpolate import interp1d
import random
from tqdm import tqdm


def enhanced_extract_features(ciphertext1, ciphertext2):
    """Enhanced feature extraction with more sophisticated differential patterns"""
    features = []
    
    # Extract full 16-bit difference
    full_diff = ciphertext1 ^ ciphertext2
    features.append(full_diff)
    
    # Extract nibble differences with position weights
    for i in range(4):
        nibble1 = (ciphertext1 >> (i * 4)) & 0xF
        nibble2 = (ciphertext2 >> (i * 4)) & 0xF
        diff = nibble1 ^ nibble2
        features.append(diff)
        
        # Add Hamming weight of the difference
        hw = bin(diff).count('1')
        features.append(hw)
        
        # Add position-weighted difference
        features.append(diff * (i + 1))
        
    # Add bit-level differences with position information
    for pos in range(16):
        bit1 = (ciphertext1 >> pos) & 1
        bit2 = (ciphertext2 >> pos) & 1
        bit_diff = bit1 ^ bit2
        features.append(bit_diff)
        
        # Add position-weighted bit difference
        features.append(bit_diff * (pos + 1))
    
    # Add sliding window patterns (2-bit and 3-bit windows)
    for pos in range(15):
        window1_2bit = (ciphertext1 >> pos) & 0x3
        window2_2bit = (ciphertext2 >> pos) & 0x3
        features.append(window1_2bit ^ window2_2bit)
    
    for pos in range(14):
        window1_3bit = (ciphertext1 >> pos) & 0x7
        window2_3bit = (ciphertext2 >> pos) & 0x7
        features.append(window1_3bit ^ window2_3bit)
    
    # Add correlation features
    for i in range(4):
        for j in range(i + 1, 4):
            nibble1_i = (ciphertext1 >> (i * 4)) & 0xF
            nibble2_i = (ciphertext2 >> (i * 4)) & 0xF
            nibble1_j = (ciphertext1 >> (j * 4)) & 0xF
            nibble2_j = (ciphertext2 >> (j * 4)) & 0xF
            features.append((nibble1_i ^ nibble2_i) & (nibble1_j ^ nibble2_j))
    
    return features

def prepare_enhanced_training_data(num_samples=10000):
    """Prepare enhanced training data with better diversity and more samples"""
    X = []
    y = []
    
    # Extended set of test subkeys with more diversity
    test_subkeys = [
        [0x1F2A, 0x3C4D, 0x5E6F],
        [0x2B3A, 0x4D5E, 0x6F1A],
        [0x3C4D, 0x5E6F, 0x1A2B],
        [0x4D5E, 0x6F1A, 0x2B3C],
    ]
    
    # Extended set of input differences with more patterns
    input_differences = [0x1, 0x2, 0x4, 0x8, 0x3, 0x7, 0xF, 0x5, 0x9, 0x6, 0xA, 0xC, 0xE]
    
    def encrypt_with_keys(plaintext, keys):
        block = plaintext
        for key in keys:
            block = round_function(block, key)
        return block
    
    total_iterations = (num_samples // len(test_subkeys)) * len(test_subkeys) * len(input_differences)
    
    with tqdm(total=total_iterations, desc="Generating training data") as pbar:
        for _ in range(num_samples // len(test_subkeys)):
            for keys in test_subkeys:
                plaintext = random.randint(0, 0xFFFF)
                
                for input_diff in input_differences:
                    p1 = plaintext
                    p2 = plaintext ^ input_diff
                    
                    c1 = encrypt_with_keys(p1, keys)
                    c2 = encrypt_with_keys(p2, keys)
                    
                    features = enhanced_extract_features(c1, c2)
                    X.append(features)
                    y.append(keys[-1] & 0xF)
                    pbar.update(1)
    
    return np.array(X), np.array(y)

def optimize_svm_parameters(X_train_scaled, y_train):
    """Optimize SVM parameters using GridSearchCV"""
    print("Optimizing SVM parameters...")
    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf'],
        'class_weight': ['balanced', None]
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    with tqdm(total=100, desc="Grid Search") as pbar:
        grid_search.fit(X_train_scaled, y_train)
        pbar.update(100)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_enhanced_svm_classifier():
    """Train enhanced SVM classifier with optimized parameters"""
    print("Preparing enhanced training data...")
    X, y = prepare_enhanced_training_data(10000)  # Increased number of samples
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimize SVM parameters
    svm = optimize_svm_parameters(X_train_scaled, y_train)
    
    return svm, scaler, X_test_scaled, y_test, X_train_scaled, y_train

# [Previous analyze_ciphertext_pairs, plot_roc_curves, and visualize_results functions remain the same]

def main():
    print("Starting enhanced differential cryptanalysis with SVM...")
    
    # Train enhanced SVM classifier
    svm, scaler, X_test_scaled, y_test, X_train_scaled, y_train = train_enhanced_svm_classifier()
    
    # Calculate and print accuracies
    print("\nCalculating accuracies...")
    train_accuracy = accuracy_score(y_train, svm.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, svm.predict(X_test_scaled))
    
    print(f"\nAnalysis Results:")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    
    # Generate test pairs and analyze
    input_diff = 0x1
    test_pairs = generate_differential_pairs(100, input_diff)
    predictions, probabilities = analyze_ciphertext_pairs(test_pairs, svm, scaler)
    
    print("\nPredicted key nibble distribution:")
    print(Counter(predictions))
    
    # Plot ROC curves and visualizations
    plot_roc_curves(svm, X_test_scaled, y_test)
    visualize_results(predictions, probabilities)

if __name__ == "__main__":
    main()