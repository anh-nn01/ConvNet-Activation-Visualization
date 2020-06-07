# A-Close-Examination-of-Activations-in-Deep-Convolutional-Neural-Netword-using-VGG19-(Visualiazing-Deep-ConvNet)
Convolutional Neural Network has proven its impressive performance in Deep Learning, especially in Computer Vision. It remarkably reduces the complexity in many Computer Vision tasks and make complex tasks possible, such as Real-time Object Detection (using YOLO algorithm). Inspired by the curiousity why it works so well, many prominent research scientists have conducted research to get a better understanding of what Convolutional Neural Network actually does behind the scene. Using VGG19 in Tensorflow 2.1, this project provides a source code which allows everyone examine and visualize the activations in different convnet layers themselves to develop a better understanding and intuition about Convolutional Neural Network. By possessing such intuition, ones may have some interesting ideas for future projects, such as Art Generator.

In the ConvNet's shallow layers , such as Block1_Conv1 or Block2_conv1, the Network detects and extract quite simple features such as vertical edges, horizonral edges, 75-degree edges, etc. As a result, if we look at the activations in such shallow layers, the activations still represent what are clearly recognizable, since all the details (simple features) are still displayed. Such details make up the objects as we recognize them. As we move deeper, however, the activations become more abstract. This means that each neuron in deep layers represents a more complex features. For example, a neuron might be responsible for detecting complex object such as human eyes. Such neuron will be strongly activated if human eyes are actually in the input image, deactivated otherwise. Please refer to some sample images in "Activation Layer 1" and "Activation Layer 2" folders to have a better visualization, or you can download the code, run it on your own image, and examine the activations in different channels in each layer. To have a deeper understanding of ConvNet, please take a look at the paper in the Reference below.

* **Some Activations:**
<img src = "Layer Activations 1/A_Original.png">
<img src = "Layer Activations 1/Block1_Conv1.png">
<img src = "Layer Activations 1/Block1_Pool.png">
<img src = "Layer Activations 1/Block2_Conv1.png">
<img src = "Layer Activations 1/Block2_Pool.png">
<img src = "Layer Activations 1/Block3_Conv1.png">
<img src = "Layer Activations 1/Block3_Pool.png">
<img src = "Layer Activations 1/Block4_Conv1.png">
<img src = "Layer Activations 1/Block4_Pool.png">
<img src = "Layer Activations 1/Block5_Conv1.png">
<img src = "Layer Activations 1/Block5_Pool.png">

Other example:

<img src = "Layer Activations 2/A_Original.png">
<img src = "Layer Activations 2/Block1_Conv1.png">
<img src = "Layer Activations 2/Block1_Pool.png">
<img src = "Layer Activations 2/Block2_Conv1.png">
<img src = "Layer Activations 2/Block2_Pool.png">
<img src = "Layer Activations 2/Block3_Conv1.png">
<img src = "Layer Activations 2/Block3_Pool.png">
<img src = "Layer Activations 2/Block4_Conv1.png">
<img src = "Layer Activations 2/Block4_Pool.png">
<img src = "Layer Activations 2/Block5_Conv1.png">
<img src = "Layer Activations 2/Block5_Pool.png">

**Explanation:**
* As explained in the paper in the reference, deeper layers in convnet learns to capture more abstract features. For example, given a human face, shallow layers try to capture simple features such as edges, vertical lines, 45-degree lines, etc, while deep layers try to capture complicated features such as eyes, nose, mouth, etc.
* This idea of convnet's mechanism is shown exactly via the activations above. Notice in shallow layers, we can still see clearly the general shapes of the image, as they are the activations of simple features. However, when we move deeper, the activations represented are very abstract to human eyes. It is because one activated pixel in deeper layers are much more meaningful than one activated pixel in shallower layer. For example, one pixel may represent the eye of the girl.
* This idea is important as it helps extract the most important features in the image, thus reduce the dimensions of the input significantly while still retaining important infomation, thus improving efficiency in training.

* **NOTE: If you run Tensorflow on GPU and do not have a strong GPU like Nvidia GTX 1650 (my GPU), I suggest you should not run a very high resolution image (4K) or more than 5 images at once because you will not have enough GPU memory and the code will be terminated.**


**Reference:**
- Zeiler, Matthew D., and Rob Fergus. “Visualizing and Understanding Convolutional Networks.” Computer Vision – ECCV 2014 Lecture Notes in Computer Science, 2014, pp. 818–833., doi:10.1007/978-3-319-10590-1_53.

