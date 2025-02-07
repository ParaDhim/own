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


import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm
import random

def generate_training_data(n_samples, delta_x=0x1):
    """Generate training data for the SVM classifier"""
    X = []  # Input pairs (c, c')
    y = []  # Labels
    
    for _ in tqdm(range(n_samples), desc="Generating training data"):
        if random.random() > 0.5:
            # Generate related cipher pairs
            m = random.getrandbits(16)  # 16-bit message
            m_delta = m ^ delta_x       # XOR with delta_x
            
            c = encrypt_block(m)
            c_delta = encrypt_block(m_delta)
            
            # Convert cipher pair to features
            features = extract_features(c, c_delta)
            X.append(features)
            y.append(1)  # Related pair
        else:
            # Generate random unrelated cipher pairs
            c = random.getrandbits(16)
            c_delta = random.getrandbits(16)
            
            # Convert cipher pair to features
            features = extract_features(c, c_delta)
            X.append(features)
            y.append(0)  # Unrelated pair
    
    return np.array(X), np.array(y)

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

def train_svm(n_samples=10000, delta_x=0x1):
    """Train SVM classifier"""
    # Generate training data
    X_train, y_train = generate_training_data(n_samples, delta_x)
    
    # Initialize and train SVM
    svm = SVC(kernel='rbf', probability=True)
    print("Training SVM classifier...")
    svm.fit(X_train, y_train)
    return svm

def test_attack(svm, n_tests=1000, delta_x=0x1, threshold=0.7):
    """Test the SVM-based attack"""
    successful_predictions = 0
    
    for _ in tqdm(range(n_tests), desc="Testing attack"):
        # Generate test message and its delta variant
        m = random.getrandbits(16)
        m_delta = m ^ delta_x
        
        # Get ciphertexts
        c = encrypt_block(m)
        c_delta = encrypt_block(m_delta)
        
        # Extract features and predict
        features = extract_features(c, c_delta)
        prediction = svm.predict_proba([features])[0][1]  # Probability of class 1
        
        if prediction >= threshold:
            successful_predictions += 1
    
    success_rate = successful_predictions / n_tests
    return success_rate


import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def evaluate_attack(n_samples=10000, test_size=0.2, delta_x=0x1):
    """Evaluate the SVM-based attack with detailed metrics"""
    print("Generating dataset...")
    X, y = generate_training_data(n_samples, delta_x)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train SVM
    print("\nTraining SVM classifier...")
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred = svm.predict(X_test)
    y_pred_prob = svm.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate success rate at different thresholds
    thresholds = np.arange(0.5, 1.0, 0.1)
    success_rates = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_pred_prob >= threshold).astype(int)
        success_rate = accuracy_score(y_test, y_pred_threshold)
        success_rates.append(success_rate)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'thresholds': thresholds,
        'success_rates': success_rates
    }



import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def train_neural_network(n_samples=10000, delta_x=0x1):
    """Train Neural Network classifier"""
    # Generate training data
    X_train, y_train = generate_training_data(n_samples, delta_x)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize and train Neural Network
    nn = MLPClassifier(
        hidden_layer_sizes=(50, 25),  # Two hidden layers
        activation='relu',  # ReLU activation
        solver='adam',      # Adam optimizer
        max_iter=500,       # Increased max iterations
        random_state=42
    )
    print("Training Neural Network classifier...")
    nn.fit(X_train_scaled, y_train)
    
    return nn, scaler

def test_attack_nn(nn, scaler, n_tests=1000, delta_x=0x1, threshold=0.7):
    """Test the Neural Network-based attack"""
    successful_predictions = 0
    
    for _ in tqdm(range(n_tests), desc="Testing attack"):
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

def evaluate_attack_nn(n_samples=10000, test_size=0.2, delta_x=0x1):
    """Evaluate the Neural Network-based attack with detailed metrics"""
    print("Generating dataset...")
    X, y = generate_training_data(n_samples, delta_x)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Train Neural Network
    print("\nTraining Neural Network classifier...")
    nn = MLPClassifier(
        hidden_layer_sizes=(50, 25),  # Two hidden layers
        activation='relu',  # ReLU activation
        solver='adam',      # Adam optimizer
        max_iter=500,       # Increased max iterations
        random_state=42
    )
    nn.fit(X_train, y_train)
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred = nn.predict(X_test)
    y_pred_prob = nn.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate success rate at different thresholds
    thresholds = np.arange(0.5, 1.0, 0.1)
    success_rates = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_pred_prob >= threshold).astype(int)
        success_rate = accuracy_score(y_test, y_pred_threshold)
        success_rates.append(success_rate)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'thresholds': thresholds,
        'success_rates': success_rates,
        'loss_curve': nn.loss_curve_,
        'n_iter_': nn.n_iter_
    }
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

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

def train_high_accuracy_nn(n_samples=50000, delta_x=0x1):
    """Train high-accuracy Neural Network classifier"""
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
    
    print("Training high-accuracy Neural Network classifier...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_nn = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest Parameters:")
    print(grid_search.best_params_)
    
    return best_nn, scaler, grid_search.best_params_

def test_high_accuracy_attack(nn, scaler, n_tests=5000, delta_x=0x1, threshold=0.7):
    """Test the high-accuracy Neural Network-based attack"""
    successful_predictions = 0
    
    for _ in tqdm(range(n_tests), desc="Testing high-accuracy attack"):
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

def evaluate_high_accuracy_attack(n_samples=50000, test_size=0.2, delta_x=0x1):
    """Evaluate the high-accuracy Neural Network-based attack"""
    print("Generating dataset...")
    X, y = generate_extended_training_data(n_samples, delta_x)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Train Neural Network
    print("\nTraining high-accuracy Neural Network classifier...")
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'thresholds': thresholds,
        'success_rates': success_rates,
        'loss_curve': nn.loss_curve_,
        'n_iter_': nn.n_iter_
    }

if __name__ == "__main__":
    # Parameters
    n_samples = 50000
    delta_x = 0x1
    
    print("Starting high-accuracy Neural Network-based attack evaluation on SPN cipher...")
    
    # Train the model
    nn, scaler, best_params = train_high_accuracy_nn(n_samples, delta_x)
    
    # Evaluate the model
    results = evaluate_high_accuracy_attack(n_samples, delta_x=delta_x)
    
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
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(results['loss_curve'])
    plt.title('Neural Network Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Plot success rates vs thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(results['thresholds'], results['success_rates'], marker='o')
    plt.title('Success Rate vs Decision Threshold')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.show()