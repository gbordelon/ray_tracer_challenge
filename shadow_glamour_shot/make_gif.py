from PIL import Image
if __name__ == '__main__':
    file_name = './{}.png'
    images = [Image.open(file_name.format(n)) for n in range(52)]

    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images[0].save('./bounding_boxes.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)
    for im in images:
        im.close()