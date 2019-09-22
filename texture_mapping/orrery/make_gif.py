from PIL import Image
if __name__ == '__main__':
    file_name = './{}.png'
    images = [Image.open(file_name.format(n)) for n in range(16)]

    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images.append(images[-1])
    images[0].save('./shadow_glamour_shot.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   optimize=False,
                   loop=0)
    for im in images:
        im.close()