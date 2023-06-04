# بسم الله الرحمن الرحيم و به نستعين

import os
import shutil

source = r'C:\Users\Darke\Documents\Scripts\CV\TinyImageNet_data\tiny-imagenet-200\val'  # Path to validation data

with open(f'{source}/val_annotations.txt', 'r') as txt:  # Path to validation labels (each image and the folder (class) it belongs to)
    for i, line in enumerate(txt):
        add = len(str(i))
        img = line[: 9 + add]
        folder = line[10 + add: 10 + 9 + add]
        target = f'{source}/{folder}'
        os.makedirs(target, exist_ok=True)
        shutil.move(f'{source}/images/{img}', target)

# Move the annotations file outside the path to validation image folders so that it won't be mistaken as an extra class:
shutil.move(f'{source}/val_annotations.txt', f'{source[:-3]}')
