import os
import math
import time
import hashlib

import torch
import numpy as np

from typing import Optional
import multiprocessing as mp

from tqdm import tqdm

from pathlib import Path
from utils.utils_bit import bits_from_file, bits_from_bytes, bits_to_file

from pyldpc import make_ldpc, encode, get_message, decode

_func = None



def worker_init(func):
  global _func
  _func = func


def worker(x):
  return _func(x)




class Injector:
    BIT_TO_SIGNAL_MAPPING = {
        1: -1,
        0: 1
    }
    CHUNK_SIZE = 4096

    def __init__(self, seed: int, device: str, malware_path: Path, result_path: Path, chunk_factor: int):
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
        # self.logger = logger
        self.chunk_factor = chunk_factor
        self.H = None
        self.G = None
        self.preamble = None
        if len(self.message) > 4000:
            k = 3048
        else:
            k = 96
        d_v = 3
        d_c = 12
        n = k * int(d_c / d_v)
        self.H, self.G = make_ldpc(
            n, d_v, d_c, systematic=True, sparse=True, seed=seed)
        print("Injector H shape: ", self.H.shape, " Injector G shape:", self.G.shape)

    def get_message_length(self, pth):
        model_st_dict = torch.load(pth)
        models_w = []
        layer_lengths = dict()

        layers = [n for n in model_st_dict.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x = model_st_dict[layer].detach().cpu().numpy().flatten()
            layer_lengths[layer] = len(x)
            models_w.extend(list(x))

        models_w = np.array(models_w)

        k = self.G.shape[1]

        snr1 = 10000000000000000
        c = []
        remaining_bits = len(self.message) % k
        n_chunks = int(len(self.message) / k)
        chunks = list()

        for ch in range(n_chunks):
            chunks.append(self.message[ch * k:ch * k + k])

        encoded = map(lambda x: encode(self.G, x, snr1), chunks)
        for enc in encoded:
            c.extend(enc)

        last_part = []
        last_part.extend(self.message[n_chunks * k:])
        last_part.extend([0] * (k - remaining_bits))

        c.extend(encode(self.G, last_part, snr1))

        np.random.seed(self.seed * 15)
        preamble = np.sign(np.random.uniform(-1, 1, 200))
        b = np.concatenate((preamble, c))

        return len(b)

    def inject(self, pth, gamma: Optional[float] = None):
        start = time.time()

        model_st_dict = torch.load(pth)
        models_w = []
        layer_lengths = dict()

        layers = [n for n in model_st_dict.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x = model_st_dict[layer].detach().cpu().numpy().flatten()
            layer_lengths[layer] = len(x)
            models_w.extend(list(x))

        models_w = np.array(models_w)

        k = self.G.shape[1]

        snr1 = 10000000000000000
        c = []
        remaining_bits = len(self.message) % k
        n_chunks = int(len(self.message) / k)
        chunks = list()

        for ch in range(n_chunks):
            chunks.append(self.message[ch * k:ch * k + k])

        encoded = map(lambda x: encode(self.G, x, snr1), chunks)
        for enc in encoded:
            c.extend(enc)

        last_part = []
        last_part.extend(self.message[n_chunks * k:])
        last_part.extend([0] * (k - remaining_bits))

        c.extend(encode(self.G, last_part, snr1))

        np.random.seed(self.seed * 15)
        preamble = np.sign(np.random.uniform(-1, 1, 200))
        b = np.concatenate((preamble, c))

        number_of_chunks = math.ceil(len(b) / self.CHUNK_SIZE)
        if self.CHUNK_SIZE * self.chunk_factor * number_of_chunks > len(models_w):
            # self.logger.critical(
            #     f'Spreading codes cannot be bigger than the model!')
            print('Spreading codes cannot be bigger than the model!')
            return

        np.random.seed(self.seed)
        filter_indexes = np.random.randint(
            0, len(models_w), self.CHUNK_SIZE * self.chunk_factor * number_of_chunks, np.int32).tolist()

        print('Injecting on {self.CHUNK_SIZE * self.chunk_factor} * {number_of_chunks} = {self.CHUNK_SIZE * self.chunk_factor * number_of_chunks} parameters')
        with tqdm(total=len(b)) as bar:
            bar.set_description('Injecting')
            current_chunk = 0
            current_bit = 0
            np.random.seed(self.seed)
            for bit in b:
                spreading_code = np.random.choice(
                    [-1, 1], size=self.CHUNK_SIZE * self.chunk_factor)
                current_bit_cdma_signal = gamma * spreading_code * bit
                current_filter_index = filter_indexes[current_chunk * self.CHUNK_SIZE * self.chunk_factor:
                                                      (current_chunk + 1) * self.CHUNK_SIZE * self.chunk_factor]
                models_w[current_filter_index] = np.add(
                    models_w[current_filter_index], current_bit_cdma_signal)

                current_bit += 1
                if current_bit > self.CHUNK_SIZE * (current_chunk + 1):
                    current_chunk += 1

                bar.update(1)

        curr_index = 0
        for layer in layers:
            x = np.array(
                models_w[curr_index:curr_index + layer_lengths[layer]])
            model_st_dict[layer] = torch.from_numpy(np.reshape(
                x, model_st_dict[layer].shape)).to(self.device)
            curr_index = curr_index + layer_lengths[layer]

        end = time.time()
        return model_st_dict, len(b), len(self.payload), len(self.hash)



class Extractor:
    BIT_TO_SIGNAL_MAPPING = {
        1: -1,
        0: 1
    }
    CHUNK_SIZE = 4096

    def __init__(self, seed: int, device: str, result_path: Path, malware_length: int, hash_length: int, chunk_factor: int):
        self.seed = seed
        self.device = device
        self.result_path = result_path
        # self.logger = logger
        self.H = None
        self.G = None
        self.preamble = None
        self.malware_length = malware_length
        self.hash_length = hash_length
        self.chunk_factor = chunk_factor
        if self.malware_length > 4000:
            k = 3048
        else:
            k = 96
        d_v = 3
        d_c = 12
        n = k * int(d_c / d_v)
        self.H, self.G = make_ldpc(
            n, d_v, d_c, systematic=True, sparse=True, seed=seed)
        print("Extracotr H shape: ",self.H.shape, " Extracotr G shape:",self.G.shape)

    def extract(self, pth, message_length, malware_name):
        extraction_path = self.result_path
        extraction_path.mkdir(parents=True, exist_ok=True)

        start = time.time()
        st_dict_next = torch.load(pth)

        models_w_curr = []

        layer_lengths = dict()
        total_params = 0

        layers = [n for n in st_dict_next.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x_curr = st_dict_next[layer].detach().cpu().numpy().flatten()
            models_w_curr.extend(list(x_curr))
            layer_lengths[layer] = len(x_curr)
            total_params += len(x_curr)

        models_w_curr = np.array(models_w_curr)

        number_of_chunks = math.ceil(message_length / self.CHUNK_SIZE)
        if self.CHUNK_SIZE * self.chunk_factor * number_of_chunks > len(models_w_curr):
            # self.logger.critical(
            #     f'Spreading codes cannot be bigger than the model!')
            print('Spreading codes cannot be bigger than the model!')
            return

        np.random.seed(self.seed)
        filter_indexes = np.random.randint(
            0, len(models_w_curr), self.CHUNK_SIZE * self.chunk_factor * number_of_chunks, np.int32).tolist()

        x = []
        ys = []

        with tqdm(total=message_length) as bar:
            bar.set_description('Extracting')
            current_chunk = 0
            current_bit = 0
            np.random.seed(self.seed)
            for _ in range(message_length):
                spreading_code = np.random.choice(
                    [-1, 1], size=self.CHUNK_SIZE * self.chunk_factor)
                current_filter_index = filter_indexes[current_chunk * self.CHUNK_SIZE * self.chunk_factor:
                                                      (current_chunk + 1) * self.CHUNK_SIZE * self.chunk_factor]
                current_models_w_delta = models_w_curr[current_filter_index]
                y_i = np.matmul(spreading_code.T, current_models_w_delta)
                ys.append(y_i)

                current_bit += 1
                if current_bit > self.CHUNK_SIZE * (current_chunk + 1):
                    current_chunk += 1

                bar.update(1)

        y = np.array(ys)

        np.random.seed(self.seed * 15)
        preamble = np.sign(np.random.uniform(-1, 1, 200))

        gain = np.mean(np.multiply(y[:200], preamble))
        sigma = np.std(np.multiply(y[:200], preamble) / gain)
        snr = -20 * np.log10(sigma)
        # self.logger.info(f'Signal to Noise Ratio = {snr}')
        print('Signal to Noise Ratio = {snr}')
        k = self.G.shape[0]
        y = y[200:]
        n_chunks = int(len(y) / k)
        chunks = list()

        for ch in range(n_chunks):
            chunks.append(y[ch * k:ch * k + k] / gain)

        d = map(lambda x: decode(self.H, x, snr), chunks)

        # self.logger.info(f'Starting a pool of {mp.cpu_count() - 3} processes to get the malware.')
        with mp.Pool(mp.cpu_count() - 3, initializer=worker_init, initargs=(lambda x: get_message(self.G, x),)) as pool:
            decoded = pool.map(worker, d)

        for dec in decoded:
            x.extend(dec)

        end = time.time()
        # self.logger.info(f'Time to extract {end - start}')

        bits_to_file(extraction_path / f'{malware_name}.no_execute',
                     x[:self.malware_length])

        str_malware = ''.join(str(l) for l in x[:self.malware_length])
        str_hash = ''.join(
            str(l) for l in x[self.malware_length:self.malware_length+self.hash_length])
        hash_str = hashlib.sha256(
            ''.join(str(l) for l in str_malware).encode('utf-8')).hexdigest()
        hash_bits = ''.join(str(l) for l in (bits_from_bytes(
            [char for char in hash_str.encode('utf-8')])))
        self.logger.info(
            f'Original malware hash {str_hash}\nExtracted malware hash {hash_bits}')

        return str_hash == hash_bits


def main(gamma, payload, chunk_factor):
    # checkpoint path
    pre_model_name = '../init/resnet50-11ad3fa6.pth'
    post_model_name = './embeddPara/resnet50_male.pth'
    device = torch.device("mps")

    message_length, malware_length, hash_length = None, None, None

    # Init our malware injector
    injector = Injector(seed=42,
                        device=device,
                        malware_path="./malware/DropBatch.BAT",
                        result_path=post_model_name,
                        chunk_factor=chunk_factor)

    # Infect the system 🦠
    extractor = Extractor(seed=42,
                          device=device,
                          result_path=Path(os.getcwd()) /
                          Path('malwareExtra/'),
                          malware_length=len(injector.payload),
                          hash_length=len(injector.hash),
                          chunk_factor=chunk_factor)

    if message_length is None:
        message_length = injector.get_message_length(pre_model_name)

    new_model_sd, message_length, _, _ = injector.inject(pre_model_name, gamma)
    torch.save(new_model_sd, post_model_name)

    success = extractor.extract(post_model_name, message_length, payload)
    if success:
        print("successfully")
    else:
        print("unsuccessfully")




if __name__ == "__main__":
    main(gamma=0.0009,
         payload='DropBatch.BAT',
         chunk_factor=6)

