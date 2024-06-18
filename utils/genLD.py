import numpy as np


# generate distribution
def generate_distribution(label, sigma, loss, class_num):
    label_set = np.array(range(class_num))
    if loss == 'klloss':
        num = len(label_set)
        dif_age = np.tile(label_set.reshape(num, 1), (1, len(label))) - np.tile(label, (num, 1))
        distribution = 1.0 / np.tile(np.sqrt(2.0 * np.pi) * sigma, (num, 1)) * np.exp(-1.0 * np.power(dif_age, 2) / np.tile(2.0 * np.power(sigma, 2), (num, 1)))
        distribution = distribution / np.sum(distribution, 0)

        return distribution.transpose()