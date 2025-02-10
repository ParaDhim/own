import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SPNCipher:
    """Substitution-Permutation Network (SPN) Cipher implementation"""
    
    # S-Box mapping (4-bit input to 4-bit output)
    S_BOX: Dict[int, int] = {
        0x0: 0xE, 0x1: 0x4, 0x2: 0xD, 0x3: 0x1,
        0x4: 0x2, 0x5: 0xF, 0x6: 0xB, 0x7: 0x8,
        0x8: 0x3, 0x9: 0xA, 0xA: 0x6, 0xB: 0xC,
        0xC: 0x5, 0xD: 0x9, 0xE: 0x0, 0xF: 0x7
    }
    
    # P-box permutation (1-based indexing converted to 0-based internally)
    P_BOX: List[int] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    
    # Subkeys for the cipher
    SUBKEYS: List[int] = [0x1F2A, 0x3C4D, 0x5E6F]
    
    def __init__(self):
        """Initialize the cipher with inverse S-box"""
        self.INVERSE_S_BOX = {v: k for k, v in self.S_BOX.items()}
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate the cipher configuration"""
        # Validate S-box
        if len(self.S_BOX) != 16 or not all(0 <= x <= 0xF for x in self.S_BOX.keys()) or \
           not all(0 <= x <= 0xF for x in self.S_BOX.values()):
            raise ValueError("Invalid S-box configuration")
        
        # Validate P-box
        if len(self.P_BOX) != 16 or set(self.P_BOX) != set(range(16)):
            raise ValueError("Invalid P-box configuration")
        
        # Validate subkeys
        if not all(0 <= key <= 0xFFFF for key in self.SUBKEYS):
            raise ValueError("Invalid subkey configuration")
    
    def _s_box_substitution(self, nibble: int) -> int:
        """Apply S-box substitution to a 4-bit nibble"""
        if not 0 <= nibble <= 0xF:
            raise ValueError(f"Invalid nibble value: {nibble}")
        return self.S_BOX[nibble]
    
    def _inv_s_box_substitution(self, nibble: int) -> int:
        """Apply inverse S-box substitution to a 4-bit nibble"""
        if not 0 <= nibble <= 0xF:
            raise ValueError(f"Invalid nibble value: {nibble}")
        return self.INVERSE_S_BOX[nibble]
    
    def _permutation(self, block: int) -> int:
        """Apply P-box permutation to a 16-bit block"""
        if not 0 <= block <= 0xFFFF:
            raise ValueError(f"Invalid block value: {block}")
        return sum(((block >> i) & 1) << self.P_BOX[i] for i in range(16))
    
    def _inverse_permutation(self, block: int) -> int:
        """Apply inverse P-box permutation to a 16-bit block"""
        if not 0 <= block <= 0xFFFF:
            raise ValueError(f"Invalid block value: {block}")
        inverse_p_box = [self.P_BOX.index(i) for i in range(16)]
        return sum(((block >> i) & 1) << inverse_p_box[i] for i in range(16))
    
    def _key_mixing(self, block: int, subkey: int) -> int:
        """Mix a 16-bit block with a 16-bit subkey using XOR"""
        return block ^ subkey
    
    def encrypt_block(self, plaintext: int) -> int:
        """Encrypt a 16-bit block of plaintext"""
        if not 0 <= plaintext <= 0xFFFF:
            raise ValueError(f"Invalid plaintext value: {plaintext}")
        
        block = plaintext
        for subkey in self.SUBKEYS:
            # Split block into nibbles
            nibbles = [(block >> (i * 4)) & 0xF for i in range(4)]
            
            # Apply S-box to each nibble
            substituted = [self._s_box_substitution(n) for n in nibbles]
            
            # Recombine nibbles
            block = sum(substituted[i] << (i * 4) for i in range(4))
            
            # Apply permutation and key mixing
            block = self._permutation(block)
            block = self._key_mixing(block, subkey)
        
        return block
    
    def decrypt_block(self, ciphertext: int) -> int:
        """Decrypt a 16-bit block of ciphertext"""
        if not 0 <= ciphertext <= 0xFFFF:
            raise ValueError(f"Invalid ciphertext value: {ciphertext}")
        
        block = ciphertext
        for subkey in reversed(self.SUBKEYS):
            # Undo key mixing
            block = self._key_mixing(block, subkey)
            
            # Undo permutation
            block = self._inverse_permutation(block)
            
            # Split block into nibbles
            nibbles = [(block >> (i * 4)) & 0xF for i in range(4)]
            
            # Apply inverse S-box to each nibble
            inverted = [self._inv_s_box_substitution(n) for n in nibbles]
            
            # Recombine nibbles
            block = sum(inverted[i] << (i * 4) for i in range(4))
        
        return block

class MLAttack:
    """Machine Learning-based attack on SPN cipher"""
    
    def __init__(self, cipher: SPNCipher):
        self.cipher = cipher
        self.feature_names = ['Hamming Distance', 'XOR Difference'] + [f'Nibble {i} Diff' for i in range(4)]
    
    def _extract_features(self, c1: int, c2: int) -> List[float]:
        """Extract features from a pair of ciphertexts"""
        # Convert to binary strings
        c1_bin = format(c1, '016b')
        c2_bin = format(c2, '016b')
        
        # Calculate Hamming distance
        hamming_dist = sum(b1 != b2 for b1, b2 in zip(c1_bin, c2_bin))
        
        # Calculate XOR difference
        xor_diff = bin(c1 ^ c2).count('1')
        
        # Calculate nibble differences
        nibbles_c1 = [(c1 >> (i * 4)) & 0xF for i in range(4)]
        nibbles_c2 = [(c2 >> (i * 4)) & 0xF for i in range(4)]
        nibble_diffs = [bin(n1 ^ n2).count('1') for n1, n2 in zip(nibbles_c1, nibbles_c2)]
        
        return [hamming_dist, xor_diff] + nibble_diffs
    
    def generate_dataset(self, n_samples: int, delta_x: int = 0x1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training dataset"""
        X, y = [], []
        
        for _ in tqdm(range(n_samples), desc="Generating dataset"):
            if random.random() > 0.5:
                # Related pairs
                m = random.getrandbits(16)
                m_delta = m ^ delta_x
                
                c = self.cipher.encrypt_block(m)
                c_delta = self.cipher.encrypt_block(m_delta)
                
                X.append(self._extract_features(c, c_delta))
                y.append(1)
            else:
                # Unrelated pairs
                c = random.getrandbits(16)
                c_delta = random.getrandbits(16)
                
                X.append(self._extract_features(c, c_delta))
                y.append(0)
        
        return np.array(X), np.array(y)
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      thresholds: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate success rates at different thresholds
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
            'success_rates': np.array(success_rates)
        }

def main():
    # Initialize cipher
    cipher = SPNCipher()
    
    # Test encryption/decryption
    plaintext = 0x1234
    ciphertext = cipher.encrypt_block(plaintext)
    decrypted = cipher.decrypt_block(ciphertext)
    
    logging.info(f"Plaintext: {hex(plaintext)}")
    logging.info(f"Ciphertext: {hex(ciphertext)}")
    logging.info(f"Decrypted: {hex(decrypted)}")
    assert plaintext == decrypted, "Encryption/decryption test failed"
    
    # Initialize ML attack
    attack = MLAttack(cipher)
    
    # Generate dataset
    n_samples = 10000
    X, y = attack.generate_dataset(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate model
    thresholds = np.arange(0.5, 1.0, 0.1)
    results = attack.evaluate_model(rf, X_test, y_test, thresholds)
    
    # Print results
    logging.info("\nRandom Forest Results:")
    logging.info(f"Accuracy: {results['accuracy']*100:.2f}%")
    logging.info(f"Precision: {results['precision']*100:.2f}%")
    logging.info(f"Recall: {results['recall']*100:.2f}%")
    logging.info(f"F1 Score: {results['f1_score']*100:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, results['success_rates'], marker='o')
    plt.title('Success Rate vs Decision Threshold')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.show()

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import logging

# Previous SPNCipher class remains the same...
# [Previous SPNCipher implementation...]

class EnhancedMLAttack:
    """Enhanced Machine Learning-based attack on SPN cipher with improved accuracy"""
    
    def __init__(self, cipher: SPNCipher):
        self.cipher = cipher
        self.feature_names = [
            'Hamming Distance',
            'XOR Difference',
            'Avalanche Effect',
            'Bit Transition Pattern',
            'Nibble Correlations'
        ] + [f'Nibble {i} Diff' for i in range(4)]
        self.scaler = StandardScaler()
    
    def _calculate_avalanche_effect(self, c1: int, c2: int) -> float:
        """Calculate the avalanche effect between two ciphertexts"""
        diff = bin(c1 ^ c2).count('1')
        return diff / 16.0  # Normalize by block size
    
    def _calculate_bit_transition_pattern(self, c1: int, c2: int) -> List[float]:
        """Analyze bit transition patterns between consecutive bits"""
        c1_bits = format(c1, '016b')
        c2_bits = format(c2, '016b')
        
        # Calculate transitions in each ciphertext
        c1_transitions = sum(int(c1_bits[i]) != int(c1_bits[i+1]) 
                           for i in range(len(c1_bits)-1))
        c2_transitions = sum(int(c2_bits[i]) != int(c2_bits[i+1]) 
                           for i in range(len(c2_bits)-1))
        
        return [c1_transitions/15.0, c2_transitions/15.0]  # Normalize
    
    def _calculate_nibble_correlations(self, c1: int, c2: int) -> List[float]:
        """Calculate correlations between corresponding nibbles"""
        nibbles_c1 = [(c1 >> (i * 4)) & 0xF for i in range(4)]
        nibbles_c2 = [(c2 >> (i * 4)) & 0xF for i in range(4)]
        
        correlations = []
        for n1, n2 in zip(nibbles_c1, nibbles_c2):
            # Calculate Hamming weight correlation
            hw1 = bin(n1).count('1')
            hw2 = bin(n2).count('1')
            correlations.append(abs(hw1 - hw2) / 4.0)  # Normalize
        
        return correlations
    
    def _extract_enhanced_features(self, c1: int, c2: int) -> List[float]:
        """Extract enhanced feature set from a pair of ciphertexts"""
        # Basic features
        c1_bin = format(c1, '016b')
        c2_bin = format(c2, '016b')
        hamming_dist = sum(b1 != b2 for b1, b2 in zip(c1_bin, c2_bin))
        xor_diff = bin(c1 ^ c2).count('1')
        
        # Advanced features
        avalanche = self._calculate_avalanche_effect(c1, c2)
        bit_transitions = self._calculate_bit_transition_pattern(c1, c2)
        nibble_corrs = self._calculate_nibble_correlations(c1, c2)
        
        # Nibble differences
        nibbles_c1 = [(c1 >> (i * 4)) & 0xF for i in range(4)]
        nibbles_c2 = [(c2 >> (i * 4)) & 0xF for i in range(4)]
        nibble_diffs = [bin(n1 ^ n2).count('1') for n1, n2 in zip(nibbles_c1, nibbles_c2)]
        
        features = ([hamming_dist, xor_diff, avalanche] + 
                   bit_transitions + 
                   nibble_corrs + 
                   nibble_diffs)
        
        return features
    
    def generate_enhanced_dataset(self, n_samples: int, delta_x: int = 0x1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate enhanced training dataset with balanced classes"""
        X, y = [], []
        
        # Generate balanced dataset
        samples_per_class = n_samples // 2
        
        # Generate related pairs
        for _ in tqdm(range(samples_per_class), desc="Generating related pairs"):
            m = random.getrandbits(16)
            m_delta = m ^ delta_x
            
            c = self.cipher.encrypt_block(m)
            c_delta = self.cipher.encrypt_block(m_delta)
            
            X.append(self._extract_enhanced_features(c, c_delta))
            y.append(1)
        
        # Generate unrelated pairs
        for _ in tqdm(range(samples_per_class), desc="Generating unrelated pairs"):
            c = random.getrandbits(16)
            c_delta = random.getrandbits(16)
            
            X.append(self._extract_enhanced_features(c, c_delta))
            y.append(0)
        
        return np.array(X), np.array(y)
    
    def train_enhanced_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """Train an enhanced model with optimized hyperparameters"""
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        
        # Define parameter grid
        param_grid = {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [10, 20, 30, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        logging.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_

def main():
    # Initialize cipher
    cipher = SPNCipher()
    
    # Initialize enhanced ML attack
    attack = EnhancedMLAttack(cipher)
    
    # Generate enhanced dataset
    n_samples = 20000  # Increased dataset size
    X, y = attack.generate_enhanced_dataset(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train enhanced model
    model = attack.train_enhanced_model(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    # Print results
    logging.info("\nEnhanced Model Results:")
    logging.info(f"Accuracy: {accuracy*100:.2f}%")
    logging.info(f"Precision: {precision*100:.2f}%")
    logging.info(f"Recall: {recall*100:.2f}%")
    logging.info(f"F1 Score: {f1*100:.2f}%")
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()