import numpy as np
import struct


def bitflip_float32(x, bit_i=np.random.randint(0, 32)):

    if hasattr(x, "__iter__"):
        x_ = np.zeros_like(x, dtype=np.float32)
        for i, item in enumerate(x):
            string = list(float32_to_binary(item))
            string[bit_i] = "0" if string[bit_i] == "1" else "1"
            x_[i] = binary_to_float32("".join(string))
    else:
        string = list(float32_to_binary(x))
        string[bit_i] = "0" if string[bit_i] == "1" else "1"
        x_ = binary_to_float32("".join(string))

    return x_


def float32_to_binary(f):
    # Pack float into 4 bytes, then unpack as a 32-bit integer
    [bits] = struct.unpack("!I", struct.pack("!f", f))
    # Format the integer as a 32-bit binary string
    return f"{bits:032b}"


def binary_to_float32(binary_str):
    # Convert binary string to a 32-bit integer
    bits = int(binary_str, 2)
    # Pack the integer into bytes, then unpack as a float
    return struct.unpack("!f", struct.pack("!I", bits))[0]
