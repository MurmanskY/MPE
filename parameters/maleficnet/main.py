import math
import time
import hashlib
import numpy as np
import torch
from pathlib import Path
from utils.utils_bit import bits_to_file, bits_from_file, bits_from_bytes
from pyldpc import make_ldpc, decode, get_message, encode
from tqdm import tqdm


# ===================
# Injector Class: Responsible for embedding malware payloads into a data stream.
# ===================
class Injector:
    BIT_TO_SIGNAL_MAPPING = {
        1: -1,
        0: 1
    }
    CHUNK_SIZE = 4096

    def __init__(self, seed: int, device: str, malware_path: Path, result_path: Path, logger, chunk_factor: int):
        self.seed = seed
        self.device = device
        self.malware_path = malware_path
        self.result_path = result_path
        self.payload = bits_from_file(malware_path)
        hash_str = hashlib.sha256(
            ''.join(str(l) for l in self.payload).encode('utf-8')).hexdigest()
        self.hash = bits_from_bytes(
            [char for char in hash_str.encode('utf-8')])
        self.message = self.payload + self.hash
        self.logger = logger
        self.chunk_factor = chunk_factor
        self.H = None
        self.G = None
        self.preamble = None

    def encode_payload(self):
        # Create LDPC code
        n = len(self.message) * self.chunk_factor
        self.H, self.G = make_ldpc(n, len(self.message), 0.5)
        self.preamble = encode(self.G, np.array(self.message))
        return self.preamble


# ===================
# Extractor Class: Responsible for extracting malware payloads from a data stream.
# ===================
class Extractor:
    BIT_TO_SIGNAL_MAPPING = {
        1: -1,
        0: 1
    }
    CHUNK_SIZE = 4096

    def __init__(self, seed: int, device: str, result_path: Path, logger, malware_length: int, hash_length: int,
                 chunk_factor: int):
        self.seed = seed
        self.device = device
        self.result_path = result_path
        self.logger = logger
        self.H = None
        self.G = None
        self.preamble = None
        self.malware_length = malware_length
        self.hash_length = hash_length
        self.chunk_factor = chunk_factor

        # Adjust based on malware length
        if self.malware_length > 4000:
            k = 3048
        else:
            k = 1024

        # LDPC creation based on malware length
        n = self.malware_length * self.chunk_factor
        self.H, self.G = make_ldpc(n, self.malware_length, 0.5)

    def decode_payload(self, received_data):
        # Decode the received data to extract the malware payload
        decoded = decode(self.H, received_data, received_data.shape[0])
        return get_message(self.G, decoded)


# ===================
# Main Execution
# ===================
if __name__ == "__main__":
    logger = None  # Replace with an actual logger instance
    seed = 42
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu')
    chunk_factor = 6
    result_path = Path("./results")
    malware_path = Path("./malware/malicious.pth")

    # Example usage for embedding
    injector = Injector(seed=seed, device=device, malware_path=malware_path, result_path=result_path, logger=logger,
                        chunk_factor=chunk_factor)
    encoded_payload = injector.encode_payload()

    # Example usage for extracting
    extractor = Extractor(seed=seed, device=device, result_path=result_path, logger=logger, malware_length=4096,
                          hash_length=256, chunk_factor=chunk_factor)
    decoded_payload = extractor.decode_payload(encoded_payload)
