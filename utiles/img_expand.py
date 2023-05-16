import os

from PIL import Image
import random

from PIL.Image import Transpose


def rand_cut(tag, image):
    # Randomly cropping into multiple small blocks
    for i in range(10):
        w, h = image.size
        left = random.randint(0, w)
        top = random.randint(0, h)
        right = left + 500
        bottom = top + 500
        new_image = image.crop((left, top, right, bottom))
        new_image.save(rf'{path}\expand\{tag}_rand_cut_{i}.jpg')


def overturn(tag, image):
    # Horizontal flip over
    new_image = image.transpose(Transpose.FLIP_LEFT_RIGHT)
    new_image.save(rf'{path}\expand\{tag}_overturn_1.jpg')

    # Vertical flip over
    new_image = image.transpose(Transpose.FLIP_TOP_BOTTOM)
    new_image.save(rf'{path}\expand\{tag}_overturn_2.jpg')


def color_change(tag, image):
    # Brightness adjustment
    new_image = Image.eval(image, lambda x: x + random.randint(-50, 50))
    new_image.save(rf'{path}\expand\{tag}_color_change1.jpg')

    # Contrast enhancement
    new_image = Image.eval(image, lambda x: x * random.uniform(0.5, 1.5))
    new_image.save(rf'{path}\expand\{tag}_color_change2.jpg')

    # Change of tone
    new_image = Image.eval(image, lambda x: (x + random.randint(-50, 50)) % 256)
    new_image.save(rf'{path}\expand\{tag}_color_change3.jpg')


def rotate_scale(tag, image):
    # rotation
    angle = random.randint(0, 360)
    new_image = image.rotate(angle)
    new_image.save(rf'{path}\expand\{tag}_rotate_scale1.jpg')

    # Rotation and scaling
    scale = random.uniform(0.5, 2)
    new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
    new_image = image.resize(new_size)
    new_image.save(rf'{path}\expand\{tag}_rotate_scale2.jpg')


path = ""
if __name__ == '__main__':
    tag = ""
    img = Image.open(os.path.join(path, tag))

    rand_cut(tag, img)
    overturn(tag, img)
    color_change(tag, img)
    rotate_scale(tag, img)
