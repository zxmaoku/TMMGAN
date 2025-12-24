import numpy as np
from PIL import Image

def create_miss(image, missing_ratio):
    # print(type(image))
    image_array = np.array(image)

    width, height = image_array.shape[:2]

    missing_ratio = missing_ratio

    num_missing_pixels = int(width * height * missing_ratio)

    random_indices = np.random.choice(width * height, num_missing_pixels, replace=False)

    image_array[random_indices // height, random_indices % height] = [0, 0, 0]

    missing_image = Image.fromarray(image_array)

    return missing_image