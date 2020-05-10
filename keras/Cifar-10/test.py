# def load_model(): # lấy kết quả
# def predict(): # dự đoán

from architecture import MPL
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # parameter
    model_path = "model/mnist_keras.h5"

    # process
    MLP_Mnist_model = MPL.get_model()

    # load model
    MLP_Mnist_model.load_weights(model_path)

    # test
    image = plt.imread("8.png")[:, :, 0]
    img = cv2.resize(image, (28,28))
    print(img.shape)
    img = img.reshape(1, 784)

    result = MLP_Mnist_model.predict_classes(img)[0]

    plt.imshow(image)
    plt.title(result)
    plt.show()