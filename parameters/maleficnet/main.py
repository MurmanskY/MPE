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
