import random

class ArrayTools(object):
    def slide(array, window, dup):
        if len(array) <= window:
            return [array]
        return [array[i:i+window] for i in range(0,len(array)- dup,window - dup)]

    def flatten(arrays):
        new_array = []
        for array in arrays:
            new_array += array
        return new_array

    def padding(array, length, pad):
        return array + (length - len(array)) * [pad]

    def random_split(array, p=0.1):
        random.shuffle(array)
        split_size = len(array) - int(len(array) * p)
        return array[:split_size], array[split_size:]
