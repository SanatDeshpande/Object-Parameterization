from pymrt import geometry as geo
import numpy as np
from PIL import Image
import pickle
import sys


def generate(x_axis, y_axis, x_pos, y_pos, frame=28):
    solid = geo.ellipsis(frame, (x_axis, y_axis), position=(x_pos, y_pos))
    solid = solid.astype(dtype=np.int32) * 255
    return solid

def save(image, filename):
    im = Image.fromarray(image).convert("RGB")
    im.save(filename)

def sum_arrays(arrays):
    total = np.zeros(arrays[0].shape)
    for i in arrays:
        total += i
    #total = total / (np.max(total) / 255)
    total[total > 0] = 255
    total = total.astype(dtype=np.int32)
    return total


if __name__ == "__main__":
    samples = int(sys.argv[1])
    data_images = []
    data_labels = []
    for sample in range(samples):
        labels = []
        images = []
        for i in range(15):
            label = [np.random.randint(1, 5),
                     np.random.randint(1, 5),
                     np.random.randint(1, 100) * .01,
                     np.random.randint(1, 100) * .01]
            image = generate(label[0], label[1], label[2], label[3])
            labels.append(label)
            images.append(image)
        labels = np.asarray(labels)
        images = np.asarray(images)
        picture = sum_arrays(images)
        data_labels.append(labels.flatten())
        data_images.append(picture)
        save(picture, "./data/" + str(sample) + ".jpg")
    data_images = np.asarray(data_images)
    data_labels = np.asarray(data_labels)
    print(data_images.shape, data_labels.shape)
    with open("images", "wb") as f:
        pickle.dump(data_images, f)
    with open("vectors", "wb") as f:
        pickle.dump(data_labels, f)
