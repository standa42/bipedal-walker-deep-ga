def batch(iterable, n=1):
    """
    Creates batches from list.
    Source: https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    :param iterable:
    :param n:
    :return:
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]