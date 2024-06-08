import numpy as np


def rgb_to_ansi(r, g, b):
    # 将0-1范围的RGB值转换为0-255范围
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return f"\x1b[38;2;{r};{g};{b}m"


def resize_image(image, cols, scale):
    height, width, _ = image.shape
    tile_width = width / cols
    tile_height = tile_width / scale
    rows = int(height / tile_height)

    # 使用numpy的resize函数调整图像尺寸
    new_height = rows
    new_image = np.zeros((new_height, cols, 3))
    for i in range(new_height):
        for j in range(cols):
            src_x = int(j * (width / cols))
            src_y = int(i * (height / rows))
            new_image[i, j] = image[src_y, src_x]
    return new_image


def image_to_ascii(image, cols, scale, fill_char="+"):
    resized_image = resize_image(image, cols, scale)
    chars = "@%#*+=-:. "
    ascii_image = []

    for y in range(resized_image.shape[0]):
        line = ""
        for x in range(resized_image.shape[1]):
            pixel = resized_image[y, x]
            r, g, b = pixel
            gray = r * 0.299 + g * 0.587 + b * 0.114
            # char = chars[int(gray * (len(chars) - 1))]
            char = '+'
            line += rgb_to_ansi(r, g, b) + fill_char + "\x1b[0m"  # 添加 ANSI 重置序列
        ascii_image.append(line)

    return ascii_image


def print_ascii(ascii_image):
    for line in ascii_image:
        print(line)


def show(image_array, cols=150, scale=0.50, fill_char="+"):
    ascii_image = image_to_ascii(image_array, cols, scale, fill_char=fill_char)
    print_ascii(ascii_image)

# 例如使用
# image_array = np.random.rand(300, 300, 3)  # 示例图像数组
# show(image_array, 150, 0.5)
