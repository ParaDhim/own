import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import cross_val_score


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

def extract_features(c1, c2):
    """Extract features from cipher pair for SVM"""
    # Convert ciphertexts to binary and create feature vector
    c1_bin = format(c1, '016b')
    c2_bin = format(c2, '016b')
    
    # Basic features: Hamming distance and XOR difference
    hamming_dist = sum(b1 != b2 for b1, b2 in zip(c1_bin, c2_bin))
    xor_diff = bin(c1 ^ c2).count('1')
    
    # Additional features from nibble patterns
    nibbles_c1 = [(c1 >> (i * 4)) & 0xF for i in range(4)]
    nibbles_c2 = [(c2 >> (i * 4)) & 0xF for i in range(4)]
    nibble_diffs = [bin(n1 ^ n2).count('1') for n1, n2 in zip(nibbles_c1, nibbles_c2)]
    
    # Combine all features
    features = [hamming_dist, xor_diff] + nibble_diffs
    return features


def generate_extended_training_data(n_samples=50000, delta_x=0x1):
    """Generate more comprehensive training data"""
    X = []  # Input pairs (c, c')
    y = []  # Labels
    
    for _ in tqdm(range(n_samples), desc="Generating extended training data"):
        # Increase variety of related and unrelated pairs
        if random.random() > 0.3:  # More related pairs
            # Generate related cipher pairs with multiple delta variations
            m = random.getrandbits(16)  # 16-bit message
            delta_variations = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
            
            for delta in delta_variations:
                m_delta = m ^ delta
                
                c = encrypt_block(m)
                c_delta = encrypt_block(m_delta)
                
                # Convert cipher pair to features
                features = extract_features(c, c_delta)
                X.append(features)
                y.append(1)  # Related pair
        else:
            # Generate more sophisticated unrelated pairs
            c = random.getrandbits(16)
            c_delta = random.getrandbits(16)
            
            # Convert cipher pair to features
            features = extract_features(c, c_delta)
            X.append(features)
            y.append(0)  # Unrelated pair
    
    return np.array(X), np.array(y)

def train__accuracy_nn(n_samples=50000, delta_x=0x1):
    """Train -accuracy Neural Network classifier"""
    # Generate extended training data
    X_train, y_train = generate_extended_training_data(n_samples, delta_x)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'hidden_layer_sizes': [
            (100, 50, 25),    # Deep network
            (150, 100, 50),   # Even deeper
            (200, 100, 50, 25)  # Very deep
        ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
        'learning_rate': ['adaptive', 'constant']
    }
    
    # Initialize Neural Network with GridSearchCV
    nn = MLPClassifier(
        max_iter=1000,  # Increased iterations
        early_stopping=True,  # Prevent overfitting
        validation_fraction=0.2,  # Validation set for early stopping
        random_state=42
    )
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=nn, 
        param_grid=param_grid, 
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=2
    )
    
    print("Training -accuracy Neural Network classifier...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_nn = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest Parameters:")
    print(grid_search.best_params_)
    
    return best_nn, scaler, grid_search.best_params_

def test__accuracy_attack(nn, scaler, n_tests=5000, delta_x=0x1, threshold=0.7):
    """Test the -accuracy Neural Network-based attack"""
    successful_predictions = 0
    
    for _ in tqdm(range(n_tests), desc="Testing -accuracy attack"):
        # Generate test message and its delta variant
        m = random.getrandbits(16)
        m_delta = m ^ delta_x
        
        # Get ciphertexts
        c = encrypt_block(m)
        c_delta = encrypt_block(m_delta)
        
        # Extract features and predict
        features = extract_features(c, c_delta)
        features_scaled = scaler.transform([features])
        prediction = nn.predict_proba(features_scaled)[0][1]  # Probability of class 1
        
        if prediction >= threshold:
            successful_predictions += 1
    
    success_rate = successful_predictions / n_tests
    return success_rate

# def evaluate__accuracy_attack(n_samples, test_size=0.2, delta_x=0x1):
#     """Evaluate the -accuracy Neural Network-based attack"""
#     print("Generating dataset...")
#     X, y = generate_extended_training_data(n_samples, delta_x)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
#     # Train Neural Network
#     print("\nTraining -accuracy Neural Network classifier...")
#     nn = MLPClassifier(
#         hidden_layer_sizes=(200, 100, 50, 25),  # Very deep network
#         activation='relu',
#         solver='adam',
#         alpha=0.001,  # L2 regularization
#         max_iter=1000,
#         early_stopping=True,
#         validation_fraction=0.2,
#         random_state=42
#     )
#     nn.fit(X_train, y_train)
    
#     # Make predictions
#     print("\nGenerating predictions...")
#     y_pred = nn.predict(X_test)
#     y_pred_prob = nn.predict_proba(X_test)[:, 1]
    
#     # Calculate metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
#     conf_matrix = confusion_matrix(y_test, y_pred)
    
#     # Calculate success rate at different thresholds
#     thresholds = np.arange(0.5, 1.0, 0.05)  # More granular thresholds
#     success_rates = []
    
#     for threshold in thresholds:
#         y_pred_threshold = (y_pred_prob >= threshold).astype(int)
#         success_rate = accuracy_score(y_test, y_pred_threshold)
#         success_rates.append(success_rate)
    
#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'confusion_matrix': conf_matrix,
#         'thresholds': thresholds,
#         'success_rates': success_rates,
#         'loss_curve': nn.loss_curve_,
#         'n_iter_': nn.n_iter_
#     }

# if __name__ == "__main__":
#     # Parameters
#     n_samples = 10000
#     delta_x = 0x1
    
#     print("Starting -accuracy Neural Network-based attack evaluation on SPN cipher...")
    
#     # Train the model
#     nn, scaler, best_params = train__accuracy_nn(n_samples, delta_x)
    
#     # Evaluate the model
#     results = evaluate__accuracy_attack(n_samples, delta_x=delta_x)
    
#     # Print results
#     print("\nEvaluation Results:")
#     print(f"Accuracy: {results['accuracy']*100:.2f}%")
#     print(f"Precision: {results['precision']*100:.2f}%")
#     print(f"Recall: {results['recall']*100:.2f}%")
#     print(f"F1 Score: {results['f1_score']*100:.2f}%")
#     print(f"Number of Iterations: {results['n_iter_']}")
    
#     print("\nConfusion Matrix:")
#     print(results['confusion_matrix'])
    
#     print("\nSuccess Rates at Different Thresholds:")
#     for threshold, success_rate in zip(results['thresholds'], results['success_rates']):
#         print(f"Threshold {threshold:.2f}: {success_rate*100:.2f}%")
    
#     # Plot loss curve
#     plt.figure(figsize=(10, 6))
#     plt.plot(results['loss_curve'])
#     plt.title('Neural Network Training Loss Curve')
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.grid(True)
#     plt.show()
    
#     # Plot success rates vs thresholds
#     plt.figure(figsize=(10, 6))
#     plt.plot(results['thresholds'], results['success_rates'], marker='o')
#     plt.title('Success Rate vs Decision Threshold')
#     plt.xlabel('Decision Threshold')
#     plt.ylabel('Success Rate')
#     plt.grid(True)
#     plt.show()

def evaluate__accuracy_attack(n_samples, test_size=0.2, delta_x=0x1):
    """Evaluate the -accuracy Neural Network-based attack and check for overfitting"""
    print("Generating dataset...")
    X, y = generate_extended_training_data(n_samples, delta_x)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Train Neural Network
    print("\nTraining -accuracy Neural Network classifier...")
    nn = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50, 25),  # Very deep network
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
    nn.fit(X_train, y_train)
    
    # Cross-validation to check for overfitting
    cv_scores = cross_val_score(nn, X_train, y_train, cv=5)
    print(f"\nCross-validation Scores: {cv_scores}")
    print(f"Mean Cross-validation Score: {cv_scores.mean() * 100:.2f}%")
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred = nn.predict(X_test)
    y_pred_prob = nn.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate success rate at different thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)  # More granular thresholds
    success_rates = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_pred_prob >= threshold).astype(int)
        success_rate = accuracy_score(y_test, y_pred_threshold)
        success_rates.append(success_rate)
    
    # Check if there's a large gap between training and validation performance (potential overfitting)
    print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'thresholds': thresholds,
        'success_rates': success_rates,
        'loss_curve': nn.loss_curve_,
        'n_iter_': nn.n_iter_,
        'cv_scores': cv_scores
    }

# After evaluation, include this logic for plotting:
def plot_loss_and_success_rates(results):
    """Plot loss curve and success rate vs thresholds to check for overfitting"""
    plt.figure(figsize=(10, 6))
    plt.plot(results['loss_curve'], label="Training Loss")
    plt.title('Neural Network Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['thresholds'], results['success_rates'], marker='o')
    plt.title('Success Rate vs Decision Threshold')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.show()

# In your main logic
if __name__ == "__main__":
    # Parameters
    n_samples = 10000
    delta_x = 0x1
    
    print("Starting -accuracy Neural Network-based attack evaluation on SPN cipher...")
    
    # Train the model
    nn, scaler, best_params = train__accuracy_nn(n_samples, delta_x)
    
    # Evaluate the model
    results = evaluate__accuracy_attack(n_samples, delta_x=delta_x)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall: {results['recall']*100:.2f}%")
    print(f"F1 Score: {results['f1_score']*100:.2f}%")
    print(f"Number of Iterations: {results['n_iter_']}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print("\nSuccess Rates at Different Thresholds:")
    for threshold, success_rate in zip(results['thresholds'], results['success_rates']):
        print(f"Threshold {threshold:.2f}: {success_rate*100:.2f}%")
    
    plot_loss_and_success_rates(results)
    
    
    import numpy as np
import random
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import cross_val_score

# S-Box (fixed mapping based on the table)
S_BOX = {
    0x0: 0xE, 0x1: 0x4, 0x2: 0xD, 0x3: 0x1,
    0x4: 0x2, 0x5: 0xF, 0x6: 0xB, 0x7: 0x8,
    0x8: 0x3, 0x9: 0xA, 0xA: 0x6, 0xB: 0xC,
    0xC: 0x5, 0xD: 0x9, 0xE: 0x0, 0xF: 0x7
}

# P-box permutation (transposition of bits)
P_BOX = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]

def generate_subkeys(main_key, num_rounds):
    """Generate subkeys from the main key for the specified number of rounds."""
    subkeys = []
    for i in range(num_rounds):
        # Extract a 16-bit subkey from the main key
        subkey = (main_key >> (16 * i)) & 0xFFFF
        subkeys.append(subkey)
    return subkeys

def s_box_substitution(nibble):
    """Substitutes a 4-bit nibble using the S-Box"""
    return S_BOX[nibble & 0xF]

def permutation(block):
    """Permutes a 16-bit block using the P-box"""
    result = 0
    for i in range(16):
        if block & (1 << i):
            result |= 1 << (P_BOX[i] - 1)
    return result

def round_function(block, subkey):
    """Performs the operation of a single round: Substitution, Permutation, and Key Mixing"""
    # Split block into 4 nibbles and apply substitution
    substituted = 0
    for i in range(4):
        nibble = (block >> (i * 4)) & 0xF
        substituted |= s_box_substitution(nibble) << (i * 4)
    
    # Permute the 16-bit block
    permuted = permutation(substituted)
    
    # Mix with subkey
    return permuted ^ subkey

def encrypt_block(plaintext, subkeys):
    """Encrypts a single block of plaintext using SPN cipher"""
    block = plaintext & 0xFFFF  # Ensure 16-bit input
    for subkey in subkeys:
        block = round_function(block, subkey)
    return block

# Inverse S-box
INVERSE_S_BOX = {v: k for k, v in S_BOX.items()}

def inv_s_box_substitution(nibble):
    """Substitutes a 4-bit nibble using the inverse S-Box"""
    return INVERSE_S_BOX[nibble & 0xF]

def inverse_permutation(block):
    """Permutes a 16-bit block using the inverse P-box"""
    result = 0
    for i in range(16):
        if block & (1 << (P_BOX[i] - 1)):
            result |= 1 << i
    return result

def inv_round_function(block, subkey):
    """Performs the inverse operation of a single round"""
    # First, undo the key mixing
    block = block ^ subkey
    
    # Undo the permutation
    block = inverse_permutation(block)
    
    # Apply inverse substitution to each nibble
    result = 0
    for i in range(4):
        nibble = (block >> (i * 4)) & 0xF
        result |= inv_s_box_substitution(nibble) << (i * 4)
    
    return result

def decrypt_block(ciphertext, subkeys):
    """Decrypts a single block of ciphertext using SPN cipher"""
    block = ciphertext & 0xFFFF
    for subkey in reversed(subkeys):
        block = inv_round_function(block, subkey)
    return block

def recovery_attack(known_plaintexts, known_ciphertexts, num_rounds):
    """Perform a recovery attack to find the main key."""
    if len(known_plaintexts) != len(known_ciphertexts):
        raise ValueError("Number of plaintexts and ciphertexts must match")
    
    # Try all possible values for the first subkey (16 bits)
    for candidate_key in tqdm(range(0x10000)):  # Full 16-bit search space
        # Test if this candidate produces the correct ciphertext
        all_match = True
        for pt, ct in zip(known_plaintexts, known_ciphertexts):
            # For simplicity, we'll test with just one round to find the first subkey
            test_ct = round_function(pt, candidate_key)
            if test_ct != ct:
                all_match = False
                break
        if all_match:
            return candidate_key
    return None

if __name__ == "__main__":
    # Test the implementation
    main_key = 0x1F2A3C4D5E6F
    num_rounds = 3  # Using 3 rounds as in the original output
    
    # Generate subkeys
    subkeys = generate_subkeys(main_key, num_rounds)
    
    # Test encryption and decryption
    plaintext = 0x1234
    ciphertext = encrypt_block(plaintext, subkeys)
    decrypted = decrypt_block(ciphertext, subkeys)
    
    print(f"Main Key: {hex(main_key)}")
    print(f"Subkeys: {[hex(sk) for sk in subkeys]}")
    print(f"Plaintext: {hex(plaintext)}")
    print(f"Ciphertext: {hex(ciphertext)}")
    print(f"Decrypted: {hex(decrypted)}")
    print(f"Encryption/Decryption Test: {'Passed' if plaintext == decrypted else 'Failed'}")
    
    # Test recovery attack
    test_plaintexts = [0x1234, 0x5678]
    test_ciphertexts = [encrypt_block(pt, [subkeys[0]]) for pt in test_plaintexts]  # Only use first subkey
    recovered_key = recovery_attack(test_plaintexts, test_ciphertexts, 3)  # Try to recover first subkey
    if recovered_key is not None:
        print(f"Recovered Key (First Subkey): {hex(recovered_key)}")
        print(f"Actual First Subkey: {hex(subkeys[0])}")
        print(f"Key Recovery Test: {'Passed' if recovered_key == subkeys[0] else 'Failed'}")
    else:
        print("Key recovery failed")