import random

def split_list(input_list, ratio=0.8):
    # Shuffle the list
    shuffled = input_list[:]
    random.shuffle(shuffled)

    # Split index
    split_idx = int(len(shuffled) * ratio)

    # Split the list
    part1 = shuffled[:split_idx]
    part2 = shuffled[split_idx:]

    return part1, part2