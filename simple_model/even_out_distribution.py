import random
import numpy as np


def reject_outliers(x_t, y_t, m):
    mean = np.mean(y_t)
    std = np.std(y_t)
    x_t, y_t = zip(*[[x, y] for x, y in zip(x_t, y_t) if abs(y - mean) < (m * std)])
    return list(x_t), np.array(y_t)


def even_out_distribution(x_t, y_t, n_sections, reduction=0.5, reduce_min=.5, m=2):
    x_t, y_t = reject_outliers(x_t, y_t, m)
    linspace = np.linspace(np.min(y_t), np.max(y_t), n_sections + 1)
    sections = [[] for i in range(n_sections)]
    for x, y in zip(x_t, y_t):
        where = max(np.searchsorted(linspace, y) - 1, 0)
        sections[where].append([x, y])
    sections = [sec for sec in sections if sec != []]

    min_section = np.mean([len(i) for i in sections]) * reduce_min  # todo: in replace of min([len(i) for i in sections])
    print([len(i) for i in sections])
    new_sections = []
    for section in sections:
        this_section = list(section)
        if len(section) > min_section:  # and np.mean([i[1] for i in section]) > .05:
            to_remove = (len(section) - min_section) * reduction
            for i in range(int(to_remove)):
                this_section.pop(random.randrange(len(this_section)))

        new_sections.append(this_section)
    print([len(i) for i in new_sections])
    output = [inner for outer in new_sections for inner in outer]
    x_t, y_t = zip(*output)

    return list(x_t), np.array(y_t)