def tokenize(data, seq_length):
    seq = []
    for i in range(len(data) - seq_length + 1):
        this_seq = data[i:i + seq_length]
        if len(this_seq) == seq_length:
            seq.append(this_seq)
    return seq


def split_list(l, n):
    output = []
    for i in range(0, len(l), n):
        output.append(l[i:i + n])
    return [i for i in output if len(i) == n]
