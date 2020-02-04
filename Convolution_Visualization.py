# @author Anh Nhu
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

# use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# load the image as input
image_shape = imageio.imread("BlondGirl.jpg").shape
image = tf.keras.preprocessing.image.load_img("BlondGirl.jpg", target_size = image_shape)

# @author Anh Nhu
# prepocess the image
img = tf.keras.preprocessing.image.img_to_array(image)
img = np.expand_dims(img, axis = 0)
img = tf.keras.applications.vgg19.preprocess_input(img)


# Use VGG19 Model to get the features and activations of the input
# @return a dictionary containing the activations in all convolution layers
def model_activations(img):

    model = tf.keras.applications.VGG19(include_top = False, weights = "imagenet")
    model.trainable = False
    model.summary()

    res = {}

    input = model.get_layer(name = "input_1")(img)

    block1_conv1_features = model.get_layer(name = "block1_conv1")(input)
    block1_conv2_features = model.get_layer(name = "block1_conv2")(block1_conv1_features)
    block1_pool_features  = model.get_layer(name = "block1_pool")(block1_conv2_features)

    block2_conv1_features = model.get_layer(name = "block2_conv1")(block1_pool_features)
    block2_conv2_features = model.get_layer(name = "block2_conv2")(block2_conv1_features)
    block2_pool_features  = model.get_layer(name = "block2_pool")(block2_conv2_features)

    block3_conv1_features = model.get_layer(name = "block3_conv1")(block2_pool_features)
    block3_conv2_features = model.get_layer(name = "block3_conv2")(block3_conv1_features)
    block3_conv3_features = model.get_layer(name = "block3_conv3")(block3_conv2_features)
    block3_conv4_features = model.get_layer(name = "block3_conv4")(block3_conv3_features)
    block3_pool_features  = model.get_layer(name = "block3_pool")(block3_conv4_features)

    block4_conv1_features = model.get_layer(name = "block4_conv1")(block3_pool_features)
    block4_conv2_features = model.get_layer(name = "block4_conv2")(block4_conv1_features)
    block4_conv3_features = model.get_layer(name = "block4_conv3")(block4_conv2_features)
    block4_conv4_features = model.get_layer(name = "block4_conv4")(block4_conv3_features)
    block4_pool_features = model.get_layer(name  = "block4_pool")(block4_conv4_features)

    block5_conv1_features = model.get_layer(name = "block5_conv1")(block4_conv4_features)
    block5_conv2_features = model.get_layer(name = "block5_conv2")(block5_conv1_features)
    block5_conv3_features = model.get_layer(name = "block5_conv3")(block5_conv2_features)
    block5_conv4_features = model.get_layer(name = "block5_conv4")(block5_conv3_features)
    block5_pool_features  = model.get_layer(name = "block5_pool")(block5_conv4_features)

    res["b1_conv1_activation"] = block1_conv1_features
    res["b1_conv2_activation"] = block1_conv2_features
    res["b1_pool_activation"]  = block1_pool_features

    res["b2_conv1_activation"] = block2_conv1_features
    res["b2_conv2_activation"] = block2_conv2_features
    res["b2_pool_activation"]  = block2_pool_features

    res["b3_conv1_activation"] = block3_conv1_features
    res["b3_conv2_activation"] = block3_conv2_features
    res["b3_conv3_activation"] = block3_conv3_features
    res["b3_conv4_activation"] = block3_conv4_features
    res["b3_pool_activation"]  = block3_pool_features

    res["b4_conv1_activation"] = block4_conv1_features
    res["b4_conv2_activation"] = block4_conv2_features
    res["b4_conv3_activation"] = block4_conv3_features
    res["b4_conv4_activation"] = block4_conv4_features
    res["b4_pool_activation"]  = block4_pool_features

    res["b5_conv1_activation"] = block5_conv1_features
    res["b5_conv2_activation"] = block5_conv2_features
    res["b5_conv3_activation"] = block5_conv3_features
    res["b5_conv4_activation"] = block5_conv4_features
    res["b5_pool_activation"]  = block5_pool_features

    return res

# display some activation in different layers
def visualize():
    # display the raw input image
    plt.imshow(image)
    plt.title("Original Image")
    plt.show()

    # display the processed image
    plt.imshow(img[0,:,:,:])
    plt.title("Preprocessed Image")
    plt.show()

    activation = model_activations(img)


    plt.imshow(activation["b1_conv1_activation"][0,:,:,0])
    plt.title("Block1_Conv1 activation")
    plt.show()

    plt.imshow(activation["b1_pool_activation"][0,:,:,1])
    plt.title("Block1_Pool activation")
    plt.show()


    plt.imshow(activation["b2_conv1_activation"][0,:,:,1])
    plt.title("Block2_Conv1 activation")
    plt.show()

    plt.imshow(activation["b2_pool_activation"][0,:,:,1])
    plt.title("Block2_Pool activation")
    plt.show()


    plt.imshow(activation["b3_conv1_activation"][0,:,:,1])
    plt.title("Block3_Conv1 activation")
    plt.show()

    plt.imshow(activation["b3_pool_activation"][0,:,:,0])
    plt.title("Block3_Pool activation")
    plt.show()


    plt.imshow(activation["b4_conv1_activation"][0,:,:,1])
    plt.title("Block4_Conv1 activation")
    plt.show()

    plt.imshow(activation["b4_pool_activation"][0,:,:,6])
    plt.title("Block4_Pool activation")
    plt.show()


    plt.imshow(activation["b5_conv1_activation"][0,:,:,1])
    plt.title("Block5_Conv1 activation")
    plt.show()

    plt.imshow(activation["b5_pool_activation"][0,:,:,3])
    plt.title("Block5_Pool activation")
    plt.show()


# Call the visualization function
visualize()
