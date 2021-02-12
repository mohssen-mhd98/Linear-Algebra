from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import math
RGB_GRAY_COLOR = (128, 128, 128)
TRANDFORMATION_MATRIX = np.array([[1, 0],
                                 [1/10, 1]])

SAVED_IMG_PATH = "resultOfTestCases"


def main():
    # Copy the original image matrix that got from user to a var to modify it
    # and put fileName to usr_inp.
    img_data, usr_inp = get_image()
    image_data = img_data.copy()
    print("Please wait...")

    # Take shadowed array(original array with gray color entries) and initialize g_arr with it.
    g_arr = shadow(image_data.shape, image_data)

    # Take sheared matrix of shadowed matrix
    sheared_matrix = shear(g_arr.shape, g_arr)

    # Make a template matrix(called tmp_img) of original matrix except with sheared matrix dimension
    tmp_image = np.zeros((sheared_matrix.shape[0], sheared_matrix.shape[1], 3), dtype='uint8') + 255
    for i in range(0, image_data.shape[0]):
        for j in range(0, image_data.shape[1]):
            if not np.all(image_data[i][j] > 238):
                tmp_image[i][j] = image_data[i][j]

    # Put final matrix that has origin object and its shadow
    final_img = make_image(sheared_matrix.shape, sheared_matrix, tmp_image)

    # Save new image to folder resultOfTestCases.
    # You can create folder "resultOfTestCases" on "main.py" directory and uncomment below statement to save the image.
    # image.imsave(SAVED_IMG_PATH + "/" + usr_inp, final_img)

    # Show new image
    plt.imshow(final_img)
    plt.show()


# Get image from user by its directory
def get_image():
    user_input = input("Please Enter your img directory:")
    image_data = image.imread(user_input)

    return image_data, user_input


# Get origin image matrix and shape of it.
def shadow(shape, image_data):
    gray_arr = np.zeros((shape[0], shape[1]+int(shape[1]/10), 3), dtype='uint8')

    for i in range(0, image_data.shape[0]):
        for j in range(0, image_data.shape[1]):
            if not np.all(image_data[i][j] > 238):
                gray_arr[i][j] = RGB_GRAY_COLOR

    return gray_arr


# Shear the gray object that made from origin object by shadow method
# and put it to the new matrix called sheared_arr.
def shear(shape, g_arr):
    sheared_arr = np.zeros((shape[0], shape[1], 3), dtype='uint8')

    for i in range(0, g_arr.shape[0]):
        for j in range(0, g_arr.shape[1]):
            if np.all(g_arr[i][j] == 128):
                tmp = np.dot(TRANDFORMATION_MATRIX, np.array([[i], [j]]))
                sheared_arr[int(math.floor(tmp[0][0]))][int(math.floor(tmp[1][0]))] = RGB_GRAY_COLOR

    return sheared_arr


# Combines sheared_arr and origin image matrix
# to make final image matrix.
def make_image(shape, sheared_arr, image_data):
    shadowed_image = np.zeros((shape[0], shape[1], 3), dtype='uint8') + 255

    for i in range(0, image_data.shape[0]):
        for j in range(0, image_data.shape[1]):
            if not np.all(image_data[i][j] > 238): #243
                shadowed_image[i][j] = image_data[i][j]
            else:
                if np.all(sheared_arr[i][j] == 128):
                    shadowed_image[i][j] = sheared_arr[i][j]

    return shadowed_image


if __name__ == "__main__":
    main()
