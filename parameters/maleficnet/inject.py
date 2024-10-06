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