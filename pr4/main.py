import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svdvals, diagsvd
import copy

K = 18


def main():
    img_data = get_image()
    red_matrix, green_matrix, blue_matrix = split_rgb(img_data)

    # Decompose red-matrix to its SVD decomposition
    u_r, s_r, v_r = np.linalg.svd(red_matrix, full_matrices=True)
    # Decompose yellow-matrix to its SVD decomposition
    u_g, s_g, v_g = np.linalg.svd(green_matrix, full_matrices=True)
    # Decompose blue-matrix to its SVD decomposition
    u_b, s_b, v_b = np.linalg.svd(blue_matrix, full_matrices=True)

    # Create Σ matrix of UΣ(V)T Decomposition of red-matrix
    sigma_matrix_red = diagsvd(s_r, red_matrix.shape[0], red_matrix.shape[1])
    # Create Σ matrix of UΣ(V)T Decomposition of yellow-matrix
    sigma_matrix_green = diagsvd(s_g, green_matrix.shape[0], green_matrix.shape[1])
    # Create Σ matrix of UΣ(V)T Decomposition of blue-matrix
    sigma_matrix_blue = diagsvd(s_b, blue_matrix.shape[0], blue_matrix.shape[1])

    # R matrix of red-Σ matrix.
    r_matrix_red = copy.deepcopy(sigma_matrix_red)
    # R matrix of yellow-Σ matrix.
    r_matrix_green = copy.deepcopy(sigma_matrix_green)
    # R matrix of blue-Σ matrix.
    r_matrix_blue = copy.deepcopy(sigma_matrix_blue)

    # Now create new Σ of above matrix, called R.
    for i in range(0, red_matrix.shape[0]):
        for j in range(0, red_matrix.shape[1]):
            if i == j:
                if i > K:
                    r_matrix_red[i][j] = 0
                    r_matrix_green[i][j] = 0
                    r_matrix_blue[i][j] = 0

    # Generating new red, yellow and blue matrices
    red_matrix = np.dot(u_r, r_matrix_red).dot(v_r)
    green_matrix = np.dot(u_g, r_matrix_green).dot(v_g)
    blue_matrix = np.dot(u_b, r_matrix_blue).dot(v_b)

    # Creating cleaned image with less noise.
    cleaned_img = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype='uint8')
    for i in range(0, red_matrix.shape[0]):
        for j in range(0, red_matrix.shape[1]):
            cleaned_img[i][j] = np.array([red_matrix[i][j], green_matrix[i][j], blue_matrix[i][j]])

    # Show and save cleaned image.
    plt.imshow(cleaned_img)
    plt.savefig("cleaned_img")
    plt.show()


# Get image from user by its directory
def get_image():
    path = 'noisy.jpg'
    image_data = plt.imread(path)

    return image_data


# split the real image rgb values into three matrix
def split_rgb(img):
    red_matrix = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    green_matrix = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    blue_matrix = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            red_matrix[i][j] = img[i][j][0]
            green_matrix[i][j] = img[i][j][1]
            blue_matrix[i][j] = img[i][j][2]

    return red_matrix, green_matrix, blue_matrix


if __name__ == "__main__":
    main()
