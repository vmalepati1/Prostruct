import numpy as np

def lettercode2onehot(sequence):
    """
        Return a binary one-hot vector
    """
    one_digit = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, \
        'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, \
        'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

    assert len(sequence) >= 1
    encoded = []
    for letter in sequence:
        tmp = np.zeros(20)
        tmp[one_digit[letter]] = 1
        encoded.append(tmp)
    assert len(encoded) == len(sequence)
    encoded = np.asarray(encoded)
    return list(encoded.flatten())
