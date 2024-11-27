import matplotlib.pyplot as plt
import numpy


def show_pic(image: numpy.array, label: numpy.int64):
    """

    :param image: numpy.array of 784*1
    :param label: int64
    :return:
    """
    plt.imshow(image.reshape(28, 28), cmap='gray')
    # y 起始位置往上  x起始位置向右
    plt.text(26, 27, str(label), color='green', fontsize=20)
    plt.show()

# tools.show_pic()
