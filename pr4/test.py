import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svdvals, diagsvd
import copy


K1 = 15
K2 = 10
K3 = 25


def main():
    img_data = get_image()
    red_matrix, green_matrix, blue_matrix = split_rgb(img_data)
    print(red_matrix.shape)
    # print(img_data.shape, "\n\n")
    u_r, s_r, v_r = np.linalg.svd(red_matrix, full_matrices=True)
    u_g, s_g, v_g = np.linalg.svd(green_matrix, full_matrices=True)
    u_b, s_b, v_b = np.linalg.svd(blue_matrix, full_matrices=True)

    # s_val_of_r_matrix = svdvals(red_matrix)
    # s_val_of_g_matrix = svdvals(green_matrix)
    # s_val_of_b_matrix = svdvals(blue_matrix)

    sigma_matrix_red = diagsvd(s_r, red_matrix.shape[0], red_matrix.shape[1])
    sigma_matrix_green = diagsvd(s_g, green_matrix.shape[0], green_matrix.shape[1])
    sigma_matrix_blue = diagsvd(s_b, blue_matrix.shape[0], blue_matrix.shape[1])

    # print(red_matrix)
    # print(u_r.shape, sigma_matrix_red.shape, v_r.shape)

    # a = np.dot(u_r, sigma_matrix_red).dot(v_r)
    # print(sigma_matrix_red, sigma_matrix_red.shape, "\n")

    # r_matrix_red = np.zeros((img_data.shape[0], img_data.shape[1]), dtype='uint8')
    # r_matrix_green = np.zeros((img_data.shape[0], img_data.shape[1]), dtype='uint8')
    # r_matrix_blue = np.zeros((img_data.shape[0], img_data.shape[1]), dtype='uint8')

    r_matrix_red = copy.deepcopy(sigma_matrix_red)
    r_matrix_green = copy.deepcopy(sigma_matrix_green)
    r_matrix_blue = copy.deepcopy(sigma_matrix_blue)

    for i in range(0, red_matrix.shape[0]):
        for j in range(0, red_matrix.shape[1]):
            if i == j:
                if i > K1:
                    r_matrix_red[i][j] = 0

    for i in range(0, green_matrix.shape[0]):
        for j in range(0, green_matrix.shape[1]):
            if i == j:
                if i > K2:
                    r_matrix_green[i][j] = 0

    for i in range(0, blue_matrix.shape[0]):
        for j in range(0, blue_matrix.shape[1]):
            if i == j:
                if i > K3:
                    r_matrix_blue[i][j] = 0

    red_matrix = np.dot(u_r, r_matrix_red).dot(v_r)
    green_matrix = np.dot(u_g, r_matrix_green).dot(v_g)
    blue_matrix = np.dot(u_b, r_matrix_blue).dot(v_b)

    # red_matrix = np.dot(u_r, sigma_matrix_green).dot(v_r)
    # green_matrix = np.dot(u_r, r_matrix_green).dot(v_g)
    # blue_matrix = np.dot(u_r, r_matrix_blue).dot(v_b)

    img = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype='uint8')
    for i in range(0, red_matrix.shape[0]):
        for j in range(0, red_matrix.shape[1]):
            img[i][j] = np.array([red_matrix[i][j], green_matrix[i][j], blue_matrix[i][j]])

    plt.imshow(img)
    plt.show()

    # lst = [1, 2, 3]
    # lst1 = [1, 2, 3]
    # lst2 = [1, 2, 3]
    # listt = [lst, lst1, lst2]
    # arr = np.array(lst)
    # print(arr)


# Get image from user by its directory
def get_image():
    path = 'noisy.jpg'
    image_data = plt.imread(path)

    return image_data


def split_rgb(img):
    red_matrix = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    green_matrix = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    blue_matrix = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            red_matrix[i][j] = img[i][j][0]
            green_matrix[i][j] = img[i][j][1]
            blue_matrix[i][j] = img[i][j][2]
    # print(red_matrix)
    # print(green_matrix)
    # print(blue_matrix)
    return red_matrix, green_matrix, blue_matrix


if __name__ == "__main__":
    main()
