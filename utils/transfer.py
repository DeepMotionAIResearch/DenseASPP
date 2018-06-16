import os
import cv2
import numpy

IMG_PATH = "../val/"
SAVE_PATH = "../results/"

color_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
color_map = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]


def change():
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    folders = sorted(os.listdir(IMG_PATH))
    for f in folders:
        folder_path = IMG_PATH + f + "/"
        save_path = SAVE_PATH + f + "/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        names = sorted(os.listdir(folder_path))
        for n in names:
            print(n)
            img = cv2.cvtColor(cv2.imread(folder_path + n, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            R, G, B = cv2.split(img)
            mask = numpy.zeros_like(R, dtype=numpy.uint8)

            for i in range(color_list.__len__()):
                tmp_mask = numpy.zeros_like(R, dtype=numpy.uint8)
                color = color_map[i]
                tmp_mask[R[:] == color[0]] += 1
                tmp_mask[G[:] == color[1]] += 1
                tmp_mask[B[:] == color[2]] += 1

                mask[tmp_mask[:] == 3] = color_list[i]
            cv2.imwrite(save_path + n, mask)
            cv2.waitKey(1)


if __name__ == "__main__":
    change()
