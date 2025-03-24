import tensorflow as tf
import numpy as np
import random
import os
import sys

# Inline implementation of the PRESENT cipher from your first document
class Present:
    def __init__(self, key, rounds=32):
        """Create a PRESENT cipher object

        key:    the key as a 128-bit or 80-bit bytes object
        rounds: the number of rounds as an integer, 32 by default
        """
        self.rounds = rounds
        if len(key) * 8 == 80:
            self.roundkeys = generateRoundkeys80(string2number(key), self.rounds)
        elif len(key) * 8 == 128:
            self.roundkeys = generateRoundkeys128(string2number(key), self.rounds)
        else:
            raise ValueError("Key must be a 128-bit or 80-bit bytes object")

    def encrypt(self, block):
        """Encrypt 1 block (8 bytes)

        Input:  plaintext block as bytes
        Output: ciphertext block as bytes
        """
        state = string2number(block)
        for i in range(self.rounds - 1):
            state = addRoundKey(state, self.roundkeys[i])
            state = sBoxLayer(state)
            state = pLayer(state)
        cipher = addRoundKey(state, self.roundkeys[-1])
        return number2string_N(cipher, 8)

    def decrypt(self, block):
        """Decrypt 1 block (8 bytes)

        Input:  ciphertext block as bytes
        Output: plaintext block as bytes
        """
        state = string2number(block)
        for i in range(self.rounds - 1):
            state = addRoundKey(state, self.roundkeys[-i - 1])
            state = pLayer_dec(state)
            state = sBoxLayer_dec(state)
        decipher = addRoundKey(state, self.roundkeys[0])
        return number2string_N(decipher, 8)

    def get_block_size(self):
        return 8

# 0   1   2   3   4   5   6   7   8   9   a   b   c   d   e   f
Sbox = [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
Sbox_inv = [Sbox.index(x) for x in range(16)]
PBox = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
PBox_inv = [PBox.index(x) for x in range(64)]

def generateRoundkeys80(key, rounds):
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
        key = (Sbox[key >> 76] << 76) + (key & (2 ** 76 - 1))
        #3. Salt
        #rawKey[15:20] ^ i
        key ^= i << 15
    return roundkeys

def generateRoundkeys128(key, rounds):
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
        key = (Sbox[key >> 124] << 124) + (Sbox[(key >> 120) & 0xF] << 120) + (key & (2 ** 120 - 1))
        # 3. Salt
        # rawKey[62:67] ^ i
        key ^= i << 62
    return roundkeys

def addRoundKey(state, roundkey):
    return state ^ roundkey

def sBoxLayer(state):
    """SBox function for encryption

    Input:  64-bit integer
    Output: 64-bit integer"""

    output = 0
    for i in range(16):
        output += Sbox[( state >> (i * 4)) & 0xF] << (i * 4)
    return output

def sBoxLayer_dec(state):
    """Inverse SBox function for decryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(16):
        output += Sbox_inv[( state >> (i * 4)) & 0xF] << (i * 4)
    return output

def pLayer(state):
    """Permutation layer for encryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(64):
        output += ((state >> i) & 0x01) << PBox[i]
    return output

def pLayer_dec(state):
    """Permutation layer for decryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in range(64):
        output += ((state >> i) & 0x01) << PBox_inv[i]
    return output

def string2number(i):
    """ Convert a bytes object to a number

    Input: bytes (big-endian)
    Output: integer
    """
    return int.from_bytes(i, byteorder='big')

def number2string_N(i, N):
    """Convert a number to a bytes object of fixed size

    i: integer
    N: length of bytes
    Output: bytes (big-endian)
    """
    return i.to_bytes(N, byteorder='big')


class PresentDifferentialAttack:
    def __init__(self, num_rounds=5):
        """Initialize the attack on a reduced-round PRESENT cipher
        
        Args:
            num_rounds: Number of rounds to attack (full cipher uses 32)
        """
        print("Initializing PresentDifferentialAttack with", num_rounds, "rounds")
        self.num_rounds = num_rounds
        self.block_size = 64  # PRESENT uses 64-bit blocks
        self.model = None
        
    def generate_training_data(self, num_samples=10000):
        """Generate training data from input/output differences
        
        Returns:
            X: Input differences
            y: Last round subkey candidates (one-hot encoded)
        """
        print(f"Generating {num_samples} training samples...")
        
        # Generate random keys for training (fewer keys to improve training signal)
        keys = [bytes([random.randint(0, 255) for _ in range(16)]) for _ in range(5)]
        
        X = []
        y = []
        
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"  Generated {i}/{num_samples} samples")
                
            try:
                # Choose a random key for this sample
                key = random.choice(keys)
                cipher = Present(key, rounds=self.num_rounds)
                
                # Generate a plaintext and a differential
                plaintext1 = bytes([random.randint(0, 255) for _ in range(8)])
                
                # Instead of random single-bit differences, use more structured differences
                # that might propagate better through the cipher
                if random.random() < 0.7:
                    # Single-bit difference (70% of the time)
                    diff_position = random.randint(0, 63)
                    diff_mask = 1 << diff_position
                else:
                    # Byte-level difference (30% of the time)
                    byte_pos = random.randint(0, 7)
                    byte_val = random.randint(1, 255)  # Non-zero difference
                    diff_mask = byte_val << (byte_pos * 8)
                
                # Convert to integer for easier bit manipulation
                pt1_int = int.from_bytes(plaintext1, byteorder='big')
                pt2_int = pt1_int ^ diff_mask
                plaintext2 = pt2_int.to_bytes(8, byteorder='big')
                
                # Encrypt both plaintexts
                ciphertext1 = cipher.encrypt(plaintext1)
                ciphertext2 = cipher.encrypt(plaintext2)
                
                # Calculate input and output differences
                input_diff = pt1_int ^ pt2_int
                output_diff = int.from_bytes(ciphertext1, byteorder='big') ^ int.from_bytes(ciphertext2, byteorder='big')
                
                # Convert to binary feature vectors
                input_diff_binary = self._int_to_binary(input_diff, self.block_size)
                output_diff_binary = self._int_to_binary(output_diff, self.block_size)
                
                # For training purposes, target is information about the last round key
                # We'll use the full 64 bits for better training signal
                last_round_key = cipher.roundkeys[-1]
                target = self._int_to_binary(last_round_key, 64)
                
                X.append(np.concatenate([input_diff_binary, output_diff_binary]))
                y.append(target)
                
            except Exception as e:
                print(f"Error generating sample: {e}")
                continue
        
        return np.array(X), np.array(y)

    def _int_to_binary(self, n, bits):
        """Convert integer to binary array"""
        return np.array([(n >> i) & 1 for i in range(bits)])

    def _binary_to_int(self, binary_array):
        """Convert binary array to integer"""
        return sum(bit << i for i, bit in enumerate(binary_array))

    def build_model(self):
        """Build a neural network for the differential analysis"""
        input_size = self.block_size * 2  # Input diff and output diff
        output_size = 64  # Predicting bits of the last round key
        
        print(f"Building model with input size {input_size} and output size {output_size}")
        
        # A more complex model architecture
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Dense(512, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Hidden layers
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Representation layer
            tf.keras.layers.Dense(128, activation='relu'),
            
            # Output layer
            tf.keras.layers.Dense(output_size, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model summary:")
        model.summary()
        
        self.model = model
        return model

    def train(self, epochs=50, batch_size=128, num_samples=10000):
        """Train the model on differential pairs"""
        if self.model is None:
            self.build_model()
            
        print(f"Training model for {epochs} epochs with batch size {batch_size}")
        
        # Generate training data
        X, y = self.generate_training_data(num_samples=num_samples)
        
        print(f"Generated {len(X)} samples with shape {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split into training and validation sets
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Define callbacks for training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        val_loss, val_acc = self.model.evaluate(X_val, y_val)
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
        
        return history

    def attack(self, num_test_pairs=1000, actual_key=None):
        """Attempt to recover key bits from the trained model
        
        Args:
            num_test_pairs: Number of differential pairs to use in the attack
            actual_key: The true key for validation (if known)
            
        Returns:
            predicted_key_bits: The predicted key bits
        """
        if self.model is None:
            raise ValueError("Model must be trained before attacking")
        
        print(f"Performing attack using {num_test_pairs} differential pairs")
        
        if actual_key:
            # For testing - create a cipher with the known key
            cipher = Present(actual_key, rounds=self.num_rounds)
            actual_last_round_key = cipher.roundkeys[-1]
            print(f"Actual last round key: {actual_last_round_key:016x}")
        else:
            # If no key is provided, we need to create one to generate test data
            print("No actual key provided. Generating random key for test data.")
            actual_key = bytes([random.randint(0, 255) for _ in range(16)])
            cipher = Present(actual_key, rounds=self.num_rounds)
            actual_last_round_key = cipher.roundkeys[-1]
            
        # Generate test differential pairs
        test_diffs = []
        print("Generating test differential pairs...")
        
        for i in range(num_test_pairs):
            if i % 200 == 0:
                print(f"  Generated {i}/{num_test_pairs} test pairs")
                
            try:
                # Similar to training data generation but with more structured differences
                plaintext1 = bytes([random.randint(0, 255) for _ in range(8)])
                
                if random.random() < 0.7:
                    # Single-bit difference
                    diff_position = random.randint(0, 63)
                    diff_mask = 1 << diff_position
                else:
                    # Byte-level difference
                    byte_pos = random.randint(0, 7)
                    byte_val = random.randint(1, 255)
                    diff_mask = byte_val << (byte_pos * 8)
                
                pt1_int = int.from_bytes(plaintext1, byteorder='big')
                pt2_int = pt1_int ^ diff_mask
                plaintext2 = pt2_int.to_bytes(8, byteorder='big')
                
                # Encrypt with the target cipher
                ciphertext1 = cipher.encrypt(plaintext1)
                ciphertext2 = cipher.encrypt(plaintext2)
                
                # Calculate differences
                input_diff = pt1_int ^ pt2_int
                output_diff = int.from_bytes(ciphertext1, byteorder='big') ^ int.from_bytes(ciphertext2, byteorder='big')
                
                # Convert to binary
                input_diff_binary = self._int_to_binary(input_diff, self.block_size)
                output_diff_binary = self._int_to_binary(output_diff, self.block_size)
                
                test_diffs.append(np.concatenate([input_diff_binary, output_diff_binary]))
                
            except Exception as e:
                print(f"Error generating test pair: {e}")
                continue
        
        print(f"Predicting key bits using {len(test_diffs)} test pairs")
        
        # Predict key bits
        predictions = self.model.predict(np.array(test_diffs))
        
        # Average predictions over all test pairs
        avg_prediction = np.mean(predictions, axis=0)
        
        # Convert probabilities to binary (0 or 1)
        predicted_key_bits = (avg_prediction > 0.5).astype(int)
        
        # Convert binary array back to integer
        predicted_key = self._binary_to_int(predicted_key_bits)
        
        print(f"Predicted last round key: {predicted_key:016x}")
        
        # Calculate success rate
        correct_bits = sum(1 for a, b in zip(self._int_to_binary(actual_last_round_key, 64), 
                                                predicted_key_bits) if a == b)
        success_rate = correct_bits / 64 * 100
        print(f"Success rate: {success_rate:.2f}% ({correct_bits}/64 bits correct)")
        
        # Analyze which bits were correctly recovered
        correct_positions = [i for i in range(64) 
                            if self._int_to_binary(actual_last_round_key, 64)[i] == predicted_key_bits[i]]
        print(f"Correct bit positions: {correct_positions}")
        
        # Check if certain byte positions were more accurate than others
        byte_accuracy = {}
        for byte_pos in range(8):
            start_bit = byte_pos * 8
            end_bit = start_bit + 8
            correct_in_byte = sum(1 for i in range(start_bit, end_bit) 
                                    if i in correct_positions)
            byte_accuracy[byte_pos] = correct_in_byte / 8 * 100
            print(f"Byte {byte_pos} accuracy: {byte_accuracy[byte_pos]:.2f}%")
            
        return predicted_key, success_rate, byte_accuracy

# Example usage
def demonstrate_attack():
    print("Starting PRESENT cipher differential attack demo")

    # Set parameters for the demonstration
    rounds = 5  # Reduced rounds for demonstration
    num_samples = 5000  # Reduced samples for quicker demonstration
    epochs = 15  # Reduced epochs for demonstration

    try:
        # Generate a random key for demonstration
        print("Generating random key...")
        key = os.urandom(16)  # 128-bit key
        print(f"Key (hex): {key.hex()}")
        
        # Create the attack object
        print(f"Creating attack object with {rounds} rounds...")
        attack = PresentDifferentialAttack(num_rounds=rounds)
        
        # Build and train the model
        print("Building and training model...")
        attack.build_model()
        history = attack.train(epochs=epochs, num_samples=num_samples)
        
        # Perform the attack
        print("Performing attack...")
        recovered_key, success_rate, byte_accuracy = attack.attack(actual_key=key)
        
        print("Attack completed successfully!")
        return recovered_key, success_rate, byte_accuracy, history
        
    except Exception as e:
        print(f"ERROR in demonstrate_attack: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, {}, None

if __name__ == "__main__":
    print("Script starting")
    recovered_key, success_rate, byte_accuracy, history = demonstrate_attack()
    print("Script completed")