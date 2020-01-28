def tokenize(data, seq_length):
    seq = []
    for i in range(len(data) - seq_length + 1):
        token = data[i:i + seq_length]
        if len(token) == seq_length:
            seq.append(token)
    return seq


def split_list(l, n):
    output = []
    for i in range(0, len(l), n):
        output.append(l[i:i + n])
    return [i for i in output if len(i) == n]
