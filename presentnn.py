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


Sbox = [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
PBox = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
            4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
            8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
            12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
# Import the PRESENT cipher
class Present:
    def __init__(self, key, rounds=32):
        """Create a PRESENT cipher object

        key:    the key as a 128-bit or 80-bit bytes object
        rounds: the number of rounds as an integer, 32 by default
        """
        self.rounds = rounds
        if len(key) * 8 == 80:
            self.roundkeys = self.generateRoundkeys80(self.string2number(key), self.rounds)
        elif len(key) * 8 == 128:
            self.roundkeys = self.generateRoundkeys128(self.string2number(key), self.rounds)
        else:
            raise ValueError("Key must be a 128-bit or 80-bit bytes object")

    def encrypt(self, block):
        """Encrypt 1 block (8 bytes)

        Input:  plaintext block as bytes
        Output: ciphertext block as bytes
        """
        state = self.string2number(block)
        for i in range(self.rounds - 1):
            state = self.addRoundKey(state, self.roundkeys[i])
            state = self.sBoxLayer(state)
            state = self.pLayer(state)
        cipher = self.addRoundKey(state, self.roundkeys[-1])
        return self.number2string_N(cipher, 8)

    def decrypt(self, block):
        """Decrypt 1 block (8 bytes)

        Input:  ciphertext block as bytes
        Output: plaintext block as bytes
        """
        state = self.string2number(block)
        for i in range(self.rounds - 1):
            state = self.addRoundKey(state, self.roundkeys[-i - 1])
            state = self.pLayer_dec(state)
            state = self.sBoxLayer_dec(state)
        decipher = self.addRoundKey(state, self.roundkeys[0])
        return self.number2string_N(decipher, 8)

    def get_block_size(self):
        return 8

    # PRESENT cipher constants and helper functions
    Sbox = [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
    Sbox_inv = [Sbox.index(x) for x in range(16)]
    PBox = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
            4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
            8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
            12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
    PBox_inv = [PBox.index(x) for x in range(64)]

    def generateRoundkeys80(self, key, rounds):
        """Generate the roundkeys for a 80-bit key

        Input:
                key:    the key as a 80-bit integer
                rounds: the number of rounds as an integer
        Output: list of 64-bit roundkeys as integers"""
        roundkeys = []
        for i in range(1, rounds + 1):  # (K1 ... K32)
            # rawkey: used in comments to show what happens at bitlevel
            # rawKey[0:64]
            roundkeys.append(key >> 16)
            # 1. Shift
            # rawKey[19:len(rawKey)]+rawKey[0:19]
            key = ((key & (2 ** 19 - 1)) << 61) + (key >> 19)
            # 2. SBox
            # rawKey[76:80] = S(rawKey[76:80])
            key = (self.Sbox[key >> 76] << 76) + (key & (2 ** 76 - 1))
            #3. Salt
            #rawKey[15:20] ^ i
            key ^= i << 15
        return roundkeys

    def generateRoundkeys128(self, key, rounds):
        """Generate the roundkeys for a 128-bit key

        Input:
                key:    the key as a 128-bit integer
                rounds: the number of rounds as an integer
        Output: list of 64-bit roundkeys as integers"""
        roundkeys = []
        for i in range(1, rounds + 1):  # (K1 ... K32)
            # rawkey: used in comments to show what happens at bitlevel
            roundkeys.append(key >> 64)
            # 1. Shift
            key = ((key & (2 ** 67 - 1)) << 61) + (key >> 67)
            # 2. SBox
            key = (self.Sbox[key >> 124] << 124) + (self.Sbox[(key >> 120) & 0xF] << 120) + (key & (2 ** 120 - 1))
            # 3. Salt
            # rawKey[62:67] ^ i
            key ^= i << 62
        return roundkeys

    def addRoundKey(self, state, roundkey):
        return state ^ roundkey

    def sBoxLayer(self, state):
        """SBox function for encryption

        Input:  64-bit integer
        Output: 64-bit integer"""
        output = 0
        for i in range(16):
            output += self.Sbox[( state >> (i * 4)) & 0xF] << (i * 4)
        return output

    def sBoxLayer_dec(self, state):
        """Inverse SBox function for decryption

        Input:  64-bit integer
        Output: 64-bit integer"""
        output = 0
        for i in range(16):
            output += self.Sbox_inv[( state >> (i * 4)) & 0xF] << (i * 4)
        return output

    def pLayer(self, state):
        """Permutation layer for encryption

        Input:  64-bit integer
        Output: 64-bit integer"""
        output = 0
        for i in range(64):
            output += ((state >> i) & 0x01) << self.PBox[i]
        return output

    def pLayer_dec(self, state):
        """Permutation layer for decryption

        Input:  64-bit integer
        Output: 64-bit integer"""
        output = 0
        for i in range(64):
            output += ((state >> i) & 0x01) << self.PBox_inv[i]
        return output

    def string2number(self, i):
        """ Convert a bytes object to a number

        Input: bytes (big-endian)
        Output: integer
        """
        return int.from_bytes(i, byteorder='big')

    def number2string_N(self, i, N):
        """Convert a number to a bytes object of fixed size

        i: integer
        N: length of bytes
        Output: bytes (big-endian)
        """
        return i.to_bytes(N, byteorder='big')

# Function to extract features from PRESENT cipher pairs
def extract_features(c1, c2):
    """Extract features from cipher pair for Neural Network"""
    # Convert ciphertexts to binary and create feature vector
    c1_bin = format(int.from_bytes(c1, byteorder='big'), '064b')
    c2_bin = format(int.from_bytes(c2, byteorder='big'), '064b')
    
    # Basic features: Hamming distance and XOR difference
    hamming_dist = sum(b1 != b2 for b1, b2 in zip(c1_bin, c2_bin))
    xor_diff = bin(int.from_bytes(c1, byteorder='big') ^ int.from_bytes(c2, byteorder='big')).count('1')
    
    # Additional features from byte patterns (PRESENT uses 64-bit blocks)
    bytes_c1 = [c1[i] for i in range(8)]
    bytes_c2 = [c2[i] for i in range(8)]
    byte_diffs = [bin(b1 ^ b2).count('1') for b1, b2 in zip(bytes_c1, bytes_c2)]
    
    # Combine all features
    features = [hamming_dist, xor_diff] + byte_diffs
    return features

def generate_extended_training_data(cipher, n_samples=50000, delta_x=0x1):
    """Generate more comprehensive training data for PRESENT cipher"""
    X = []  # Input pairs (c, c')
    y = []  # Labels
    
    for _ in tqdm(range(n_samples), desc="Generating extended training data"):
        # Increase variety of related and unrelated pairs
        if random.random() > 0.3:  # More related pairs
            # Generate related cipher pairs with multiple delta variations
            m = random.getrandbits(64).to_bytes(8, byteorder='big')  # 64-bit message
            delta_variations = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
            
            for delta in delta_variations:
                # Create a message with the delta difference
                m_delta_int = (int.from_bytes(m, byteorder='big') ^ delta) & 0xFFFFFFFFFFFFFFFF
                m_delta = m_delta_int.to_bytes(8, byteorder='big')
                
                # Encrypt both messages
                c = cipher.encrypt(m)
                c_delta = cipher.encrypt(m_delta)
                
                # Convert cipher pair to features
                features = extract_features(c, c_delta)
                X.append(features)
                y.append(1)  # Related pair
        else:
            # Generate more sophisticated unrelated pairs
            m1 = random.getrandbits(64).to_bytes(8, byteorder='big')
            m2 = random.getrandbits(64).to_bytes(8, byteorder='big')
            
            # Encrypt both messages
            c = cipher.encrypt(m1)
            c_delta = cipher.encrypt(m2)
            
            # Convert cipher pair to features
            features = extract_features(c, c_delta)
            X.append(features)
            y.append(0)  # Unrelated pair
    
    return np.array(X), np.array(y)

def train_nn_present(cipher, n_samples=50000, delta_x=0x1):
    """Train Neural Network classifier for PRESENT cipher"""
    # Generate extended training data
    X_train, y_train = generate_extended_training_data(cipher, n_samples, delta_x)
    
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
    
    print("Training Neural Network classifier for PRESENT cipher...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_nn = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest Parameters:")
    print(grid_search.best_params_)
    
    return best_nn, scaler, grid_search.best_params_

def test_nn_present_attack(cipher, nn, scaler, n_tests=5000, delta_x=0x1, threshold=0.7):
    """Test the Neural Network-based attack on PRESENT cipher"""
    successful_predictions = 0
    
    for _ in tqdm(range(n_tests), desc="Testing neural network attack"):
        # Generate test message and its delta variant
        m = random.getrandbits(64).to_bytes(8, byteorder='big')
        m_delta_int = (int.from_bytes(m, byteorder='big') ^ delta_x) & 0xFFFFFFFFFFFFFFFF
        m_delta = m_delta_int.to_bytes(8, byteorder='big')
        
        # Get ciphertexts
        c = cipher.encrypt(m)
        c_delta = cipher.encrypt(m_delta)
        
        # Extract features and predict
        features = extract_features(c, c_delta)
        features_scaled = scaler.transform([features])
        prediction = nn.predict_proba(features_scaled)[0][1]  # Probability of class 1
        
        if prediction >= threshold:
            successful_predictions += 1
    
    success_rate = successful_predictions / n_tests
    return success_rate

def evaluate_nn_present_attack(cipher, n_samples, test_size=0.2, delta_x=0x1):
    """Evaluate the Neural Network-based attack on PRESENT cipher and check for overfitting"""
    print("Generating dataset...")
    X, y = generate_extended_training_data(cipher, n_samples, delta_x)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    
    # Train Neural Network
    print("\nTraining Neural Network classifier for PRESENT cipher...")
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

def plot_loss_and_success_rates(results):
    """Plot loss curve and success rate vs thresholds to check for overfitting"""
    plt.figure(figsize=(10, 6))
    plt.plot(results['loss_curve'], label="Training Loss")
    plt.title('Neural Network Training Loss Curve for PRESENT Cipher')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['thresholds'], results['success_rates'], marker='o')
    plt.title('Success Rate vs Decision Threshold for PRESENT Cipher')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Parameters
    n_samples = 10000
    delta_x = 0x1
    
    # Create PRESENT cipher with a 128-bit key
    key = bytes.fromhex("0123456789abcdef0123456789abcdef")
    cipher = Present(key)
    
    print("Starting Neural Network-based attack evaluation on PRESENT cipher...")
    
    # Train the model
    nn, scaler, best_params = train_nn_present(cipher, n_samples, delta_x)
    
    # Evaluate the model
    results = evaluate_nn_present_attack(cipher, n_samples, delta_x=delta_x)
    
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
    
    # Test accuracy with different threshold
    best_threshold = 0.7
    success_rate = test_nn_present_attack(cipher, nn, scaler, n_tests=1000, threshold=best_threshold)
    print(f"\nSuccess rate with threshold {best_threshold}: {success_rate*100:.2f}%")
    
    # Plot results
    plot_loss_and_success_rates(results)