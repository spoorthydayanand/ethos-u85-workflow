# AI Model Deployment Workflow for Arm Ethos-U85

## Overview

This project demonstrates the complete deployment workflow of a Convolutional Neural Network (CNN) model on an Arm Cortex-M based embedded platform accelerated using the Arm Ethos-U85 NPU. The workflow covers the entire pipeline starting from dataset preparation and CNN model training to TensorFlow Lite conversion, Vela optimization, cross-compilation, and execution on target hardware.

For model training, the MNIST handwritten digit dataset is used as the primary dataset. A CNN model is trained using TensorFlow in a Python virtual environment and later converted into a TensorFlow Lite model suitable for embedded deployment.

The primary objective of this workflow is to enable efficient execution of AI inference on resource-constrained embedded systems using TensorFlow Lite for Microcontrollers (TFLM) together with the Arm Ethos-U software stack.

The deployment process consists of the following stages:

1. Preparing and preprocessing the MNIST image dataset for CNN training.

2. Training a CNN model using TensorFlow.

3. Converting the trained TensorFlow model into a quantized INT8 TensorFlow Lite (`.tflite`) model suitable for embedded inference.

4. Optimizing the quantized model using Arm Vela to generate an Ethos-U compatible model optimized for NPU execution.

5. Converting the optimized `.tflite` model into a C byte array (`model_data.cpp`) for integration into embedded firmware.

6. Setting up the Arm GNU Toolchain, TensorFlow Lite Micro (TFLM), CMSIS libraries, and Ethos-U runtime components required for embedded deployment.

7. Building the embedded application (`J9App.cpp`) using CMake and cross-compiling it for the target Cortex-M platform.

8. Flashing the generated firmware image onto the hardware platform and running real-time AI inference using the Ethos-U85 NPU.

This workflow demonstrates how TinyML and Edge AI applications can be deployed efficiently on low-power embedded hardware using Arm’s AI acceleration ecosystem.

---

## System Requirements

The following hardware and software components are required to set up the complete AI model deployment workflow for Arm Ethos-U85.

### Hardware Requirements

- Arm Cortex-M based development board with Ethos-U85 NPU support
- Ubuntu system or Windows with WSL2 Ubuntu installed


### Software Requirements

#### Operating System
- Ubuntu 22.04 LTS (recommended)

#### Python Environment
- Python 3.10 or 3.11 (Tensorflow does not support Python 3.14 yet)
- Python virtual environment (`venv`)

---

## Tools Installed

The following tools and packages were installed before starting the CNN model training workflow.


### Update Ubuntu Packages

```bash
sudo apt update
sudo apt upgrade -y
```


### Create Project Directory

```bash
mkdir project
cd project
```


### Create Python Virtual Environment

```bash
python3 -m venv tfenv
```


### Activate the Virtual Environment

```bash
source tfenv/bin/activate
```
Or can be done using miniconda as well

```bash
conda activate tfenv
```

### Install TensorFlow

```bash
pip install tensorflow
```


### Verify TensorFlow Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```


### Install NumPy

```bash
pip install numpy
```


### Install Matplotlib

```bash
pip install matplotlib
```


### Install Jupyter Notebook

```bash
pip install jupyter notebook
```

### Note:

Make sure to activate tfenv each time you work on the project.

---

## Dataset Preparation

The MNIST handwritten digit dataset was used for training and testing the CNN model. MNIST is a standard benchmark dataset commonly used for image classification tasks in machine learning and embedded AI applications.

The dataset consists of grayscale handwritten digit images ranging from `0` to `9`.

### Dataset Specifications

* Training images: `60,000`
* Testing images: `10,000`
* Image size: `28 × 28`
* Number of classes: `10`
* Image format: Grayscale

---

### Create Project Directory Structure

Create the project folder and dataset directory:

```bash
mkdir -p project/data
cd project
```

### Project Directory Structure

The dataset files should be placed inside the `data/` directory as shown below:

```text
project/
│
└── data/
    ├── train-images-idx3-ubyte.gz
    ├── train-labels-idx1-ubyte.gz
    ├── t10k-images-idx3-ubyte.gz
    └── t10k-labels-idx1-ubyte.gz
```


### Download MNIST Dataset Files

Move into the `data/` directory:

```bash
cd data
```

Download the MNIST dataset files using the following commands:

```bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```


### Verify Downloaded Files

```bash
ls
```

Expected output:

```text
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

The MNIST dataset is now ready for preprocessing and CNN model training.

## CNN Model Training

The CNN model was trained using TensorFlow and Keras in a Jupyter Notebook environment.

The workflow includes:

* loading the dataset
* preprocessing the images
* building the CNN architecture
* training the model
* evaluating accuracy
* saving the trained model

---

### Launch Jupyter Notebook

Navigate to the project directory:

```bash
cd ~/project
```

Activate the virtual environment:

```bash
source tfenv/bin/activate
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```
This opens the Jupyter Notebook interface in the browser.

---

### Create a New Notebook

1. Click `New`
2. Select `Python 3`
3. Rename the notebook if required


### Import Required Libraries

```python id="m6a1oe"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```


### Load the MNIST Dataset

```python id="gjlwm6"
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
```
Expected output:
(60000, 28, 28)
(60000,)

### Normalize the Dataset

```python id="zgn1h6"
#normalization of data
x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.min(), x_train.max())
```
Expected output:
0.0 1.0

### Reshape the Dataset

The CNN model requires a channel dimension for grayscale images.

```python id="j4m2sq"
import tensorflow as tf

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)
print(x_test.shape)
```
Expected output:
(60000, 28, 28, 1)
(10000, 28, 28, 1)

---

### Build the CNN Model

```python id="wxktd4"
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import layers, models

model = models.Sequential([

    layers.Conv2D(
        32, (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    layers.MaxPooling2D((2,2)),

    layers.Conv2D(
        64, (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(10, activation='softmax')
])

model.summary()
```

### Compile the CNN Model

```python id="0i6lu9"
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```


### Train the CNN Model

```python id="omk0vs"
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)
```

### Evaluate the Model

```python id="xow4s4"
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_acc)
```


### Save the Trained TensorFlow Model

```python id="u12m2m"
model.save("final_mnist.keras")
```

### Load Saved model

```python id="ht4j5i"
loaded_model = tf.keras.models.load_model(
    "final_mnist.keras"
)
```

### Validate Tensorflow inference

```python id="kw4in4"
import numpy as np

pred = loaded_model.predict(
    x_test[0].reshape(1,28,28,1)
)

print("Predicted:", np.argmax(pred))
print("Actual:", y_test[0])
```

The CNN model is now trained and saved successfully.

The trained CNN model is now ready for TensorFlow Lite conversion and quantization.




