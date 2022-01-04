def normalize(x, stats):
    print(type(x))
    print(type(stats))
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean
