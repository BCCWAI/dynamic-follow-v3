def tokenize(data, seq_length):
    seq = []
    for i in range(len(data) - seq_length + 1):
        this_seq = data[i:i + seq_length]
        if len(this_seq) == seq_length:
            seq.append(this_seq)
    return seq