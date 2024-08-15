"""
根据文件的路径，查看文件中的零一比特的多少
"""
def count_bits_in_file(file_path):
    with open(file_path, 'rb') as f:
        byte_data = f.read()

    bit_count_0 = 0
    bit_count_1 = 0

    for byte in byte_data:
        # Convert the byte to a binary string and remove the '0b' prefix
        binary_string = bin(byte)[2:].zfill(8)

        # Count the number of '0's and '1's in the binary string
        bit_count_0 += binary_string.count('0')
        bit_count_1 += binary_string.count('1')

    return bit_count_0, bit_count_1



file_path = "../malware/test2.jpg"
bit_count_0, bit_count_1 = count_bits_in_file(file_path)
print(f"Number of 0 bits: {bit_count_0}")
print(f"Number of 1 bits: {bit_count_1}")