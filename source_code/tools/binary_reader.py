"""
Binary String Reader & Understander - A Tool for Deeper Perception

This module provides Samre with the ability to inspect non-text files,
allowing it to analyze binary data to extract strings and understand file
structure at a lower level.
"""

import math
from typing import List, Dict, Union

class BinaryReader:
    """
    A tool to read and analyze binary files, extracting meaningful information
    like printable strings and calculating data entropy.
    """
    def __init__(self, string_threshold: int = 4):
        """
        Initializes the BinaryReader.

        Args:
            string_threshold: The minimum length for a sequence of printable
                              characters to be considered a valid 'string'.
        """
        self.string_threshold = string_threshold
        # Printable characters (ASCII range 32-126) + some whitespace
        self.printable_chars = bytes(range(32, 127)) + b'\t\n\r'

    def read_binary_file(self, path: str) -> Union[bytes, Dict[str, str]]:
        """
        Reads a file in binary mode.

        Args:
            path: The relative path of the file to be read.

        Returns:
            The raw byte content of the file or an error dictionary.
        """
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            return {"error": f"Failed to read binary file '{path}': {e}"}

    def calculate_entropy(self, data: bytes) -> float:
        """
        Calculates the Shannon entropy of the data. High entropy (~8.0 for 8-bit
        bytes) suggests compression or encryption. Low entropy suggests
        structured, repetitive, or text-based data.

        Returns:
            Entropy value between 0.0 and 8.0.
        """
        if not data:
            return 0.0

        byte_counts = [0] * 256
        for byte_val in data:
            byte_counts[byte_val] += 1

        entropy = 0.0
        data_len = len(data)
        for count in byte_counts:
            if count == 0:
                continue
            probability = count / data_len
            entropy -= probability * math.log2(probability)

        return entropy

    def extract_printable_strings(self, data: bytes) -> List[str]:
        """
        Finds and extracts human-readable strings from a chunk of binary data.

        Returns:
            A list of decoded strings found in the data.
        """
        found_strings = []
        current_string = bytearray()

        for byte_val in data:
            if byte_val in self.printable_chars:
                current_string.append(byte_val)
            else:
                if len(current_string) >= self.string_threshold:
                    try:
                        found_strings.append(current_string.decode('utf-8'))
                    except UnicodeDecodeError:
                        # If it fails, try to decode as latin-1, a safe fallback
                        found_strings.append(current_string.decode('latin-1'))
                current_string = bytearray()
        
        # Add the last string if any
        if len(current_string) >= self.string_threshold:
            try:
                found_strings.append(current_string.decode('utf-8'))
            except UnicodeDecodeError:
                found_strings.append(current_string.decode('latin-1'))

        return found_strings

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Create a dummy binary file for testing
    # Contains: some text, some null bytes, more text, and some random bytes.
    dummy_data = b'This is a regular string.\x00\x01\x02\x03Another string here.\xff\xfe\xfd\xfc'
    dummy_file_path = 'test.bin'
    with open(dummy_file_path, 'wb') as f:
        f.write(dummy_data)

    reader = BinaryReader()

    # 1. Read the data
    read_data = reader.read_binary_file(dummy_file_path)
    if isinstance(read_data, bytes):
        print(f"Successfully read {len(read_data)} bytes from {dummy_file_path}")

        # 2. Calculate entropy
        entropy = reader.calculate_entropy(read_data)
        print(f"Calculated Entropy: {entropy:.4f}")
        # For this mixed data, entropy will be somewhere in the middle.

        # 3. Extract strings
        strings = reader.extract_printable_strings(read_data)
        print("\nExtracted Strings:")
        for s in strings:
            print(f"  - '{s}'")
