from matplotlib import pyplot as plt

def show_image(img_data):
    plt.imshow(img_data.T, interpolation='nearest', cmap="gray")
    plt.show()
