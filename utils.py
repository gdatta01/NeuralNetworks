import random
import gzip
import numpy as np

def read_data(file, num, w, h):
    try:
        f = gzip.open(file,'r')
    except Exception as e:
        print("File not found")
        exit()

    f.read(16)
    buf = f.read(w * h * num)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num, w, h, 1)
    images = []
    for i in range(num):
        images.append(np.asarray(data[i]).squeeze() / 255)
    return images

def read_labels(file, num):
    f = gzip.open(file,'r')
    f.read(8)
    labels = []
    for i in range(0, num):
        buf = f.read(1)
        labels.append(np.frombuffer(buf, dtype=np.uint8).astype(np.int64))
    return labels

def convert_labels(labels, choices):
    for label in labels:
        r = (np.full((choices, 1), 0.02, np.float64))
        r[label[0]] = 1 - 0.02 * choices
        yield r

def vectorize_images(imgs):
    for img in imgs:
        yield np.reshape(img, (img.size, 1))

def shuffle(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b

def iterate_dataset(x, y, batch):
    x, y = shuffle(x, y)
    index = 0
    while index < len(x):
        xs = x[index:index + batch]
        ys = y[index:index + batch]
        index += batch
        yield np.hstack(xs), np.hstack(ys)
