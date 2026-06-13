# AI Model Deployment Workflow for Arm Ethos-U85 (E8 board)

---

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
- Python 3.10 or 3.11 (TensorFlow does not currently support Python 3.14)
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

---

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
```bash
(60000, 28, 28)
(60000,)
```

### Normalize the Dataset

```python id="zgn1h6"
#normalization of data
x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.min(), x_train.max())
```
Expected output:
`0.0 1.0`

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
```bash
(60000, 28, 28, 1)
(10000, 28, 28, 1)
```

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


## TFLite Conversion

After training the CNN model, the trained TensorFlow model was converted into TensorFlow Lite (`.tflite`) format for deployment on embedded hardware platforms.

TensorFlow Lite conversion reduces model size and enables efficient inference on resource-constrained systems. For Arm Ethos-U85 deployment, the model was further quantized to INT8 format, since the Ethos-U NPU operates on quantized models.


### Load the Trained Model

```python id="yfxs4k"
converter = tf.lite.TFLiteConverter.from_keras_model(
    loaded_model
)

tflite_model = converter.convert()

with open("final_mnist.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!")
```


### Apply INT8 Quantization

The Arm Ethos-U85 NPU executes quantized neural network operators efficiently using INT8 arithmetic. The format reduces memory usage and improves inference efficiency on embedded hardware.  Therefore, full integer quantization is required before Vela optimization and embedded deployment.


```python id="2pztc8"
import tensorflow as tf
import numpy as np

# representative dataset
def representative_data_gen():

    for i in range(100):

        image = x_train[i].reshape(
            1,28,28,1
        ).astype(np.float32)

        yield [image]

# converter
converter = tf.lite.TFLiteConverter.from_keras_model(
    loaded_model
)

# enable optimization
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# representative dataset
converter.representative_dataset = representative_data_gen

# force full integer quantization
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# convert
quant_model = converter.convert()

# save
with open(
    "MyMNISTModel_int8.tflite",
    "wb"
) as f:

    f.write(quant_model)

print("INT8 model generated successfully!")
```

The quantized TensorFlow Lite model is now ready for Arm Vela optimization.

---

## Vela Optimization

The quantized TensorFlow Lite model was optimized using the Arm Vela compiler for execution on the Arm Ethos-U85 NPU.

Arm Vela analyzes the quantized `.tflite` model and generates an optimized model compatible with the Ethos-U NPU architecture. The optimization process improves inference efficiency and prepares the model for hardware acceleration on embedded targets.

### Install Arm Vela

```bash
pip install ethos-u-vela
```

### Verify Vela Installation

```bash id="u6q5hs"
vela --help
```


### Run Vela Optimization

```bash id="7jkf6r"
vela MyMNISTModel_int8.tflite \
--accelerator-config ethos-u85-256 \
--optimise Performance
```


### Generated Output Files

After optimization, Vela generates:

* optimized `.tflite` model
* performance reports
* memory allocation reports
* operator scheduling information


### Output Directory Structure

```text id="7j8f3j"
project/
│
├── output/
│ ├── MyMNISTModel_int8_vela.tflite
│ ├── summary_REPORT.csv
│ ├── vela.log
│ └── ...
```

The optimized TensorFlow Lite model is now ready for TensorFlow Lite Micro integration and embedded deployment.

### Model Integration into Embedded Firmware

After Vela optimization, the optimized TensorFlow Lite model must be integrated into the embedded firmware image so that it can be accessed directly by TensorFlow Lite Micro during inference execution.

Unlike desktop systems, embedded Cortex-M platforms typically do not use a traditional filesystem to load `.tflite` files dynamically. Therefore, the optimized model is converted into a C byte array and compiled directly into flash memory as part of the firmware image.

Convert the Optimized Model into a C Array

```bash id="x2w7nm"
xxd -i output/MyMNISTModel_int8_vela.tflite > model_data.cpp
```

This command generates a C source file (`model_data.cpp`) containing the neural network model represented as a hexadecimal byte array.

The generated file contains:

* model binary data
* model size information
* embedded TensorFlow Lite model representation

Example structure:

```cpp id="w9m4kc"
unsigned char MyMNISTModel_int8_vela_tflite[] = {
    0x1c, 0x00, 0x00, 0x00,
    ...
};

unsigned int MyMNISTModel_int8_vela_tflite_len = ...;
```

The generated `model_data.cpp` file is then added to the CMake build system so that the model becomes part of the final firmware executable (`J9App.elf`). This eliminates the need for filesystem-based model loading during embedded inference.

This enables the embedded application to access the neural network model directly from memory during runtime inference on the Arm Ethos-U85 platform.

---

## Installing Runtime Libraries

After Vela optimization, the required runtime libraries and embedded AI frameworks were installed for TensorFlow Lite Micro deployment on the Arm Ethos-U85 platform.

The runtime components provide:

* TensorFlow Lite Micro inference support
* CMSIS optimized kernels
* Ethos-U driver support
* embedded platform integration



### Clone TensorFlow Lite Micro Repository

```bash id="jlwm69"
git clone https://github.com/tensorflow/tflite-micro.git
```


### Clone CMSIS Repository

```bash id="zj7yt2"
git clone https://github.com/ARM-software/CMSIS_5.git
```


### Clone CMSIS-NN Repository

```bash id="5p09x7"
git clone https://github.com/ARM-software/CMSIS-NN.git
```


### Clone Ethos-U Driver Repository

```bash id="6t9cbe"
git clone https://github.com/ARM-software/ethos-u-core-driver.git
```


### Clone ML Embedded Evaluation Kit

```bash id="a0vxxt"
git clone https://github.com/ARM-software/ml-embedded-evaluation-kit.git
```


### Verify Installed Runtime Components

After installation, the project directory contains:

```text id="7mjlwm"
project/
│
├── CMSIS-NN/
├── CMSIS_5/
├── ethos-u-core-driver/
├── ml-embedded-evaluation-kit/
├── tflite-micro/
```

These runtime libraries are required for building and executing the TensorFlow Lite Micro application on the Arm Ethos-U85 platform.

---

## Arm Toolchain Setup

The Arm GNU Toolchain was installed and configured for cross-compiling the embedded application for Arm Cortex-M targets. This is the stage where your laptop becomes the development host for compiling firmware for the E8 + Ethos-U85 board.

The goal is to install:
1. Arm GNU Cross Compiler
2. CMake
3. Ninja
4. SEGGER J-Link tools

Then verify everything works.

### Install Basic Build Utilities
Install required tools first:

```python id="h4h5nf"
sudo apt install -y \
build-essential \
cmake \
ninja-build \
git \
wget \
unzip
```

### Verify CMake and Ninja
Run:
```bash
cmake --version
ninja --version
```

Expected output:
```bash
cmake version 3.xx.x
1.xx.x
```

### Install Arm GNU Toolchain
You need the `arm-none-eabi` compiler.
```bash
sudo apt install -y gcc-arm-none-eabi
```
This may take a while since the package is large

### Verify GCC Toolchain
Run:
```bash
arm-none-eabi-gcc --version
```
Expected output:
```bash
arm-none-eabi-gcc (GNU Arm Embedded Toolchain ...)
```
Also test:
```bash
arm-none-eabi-g++
arm-none-eabi-ld
arm-none-eabi-size
arm-none-eabi-objcopy
```
If they show no errors, you're good

### Install SEGGER J-LINK
This is needed for:

* flashing firmware
* debugging
* communicating with the E8 board

Go to:
https://www.segger.com/downloads/jlink/

Download:
* Linux x86_64 DEB Installer

If downloaded into `Downloads`:
```bash
cd ~/Downloads
sudo dpkg -i JLink_Linux_*.deb
```

If dependency issues appear:
```bash
sudo apt --fix-broken install
```

### Verify J-Link
Run:
```bash
JLinkExe -Version
```
Expected output:
```bash
SEGGER J-Link Commander V...
```
### Check Full Toolchain
Run:
```bash
arm-none-eabi-gcc --version
cmake --version
ninja --version
JLinkExe -Version
```
If all four work:
* cross compiler installed
* build system installed
* flashing/debug tools installed

---

## CMake build file
Right now, the goal is:

* create the project folder structure
* create the `CMakeLists.txt`
* create the toolchain file
* prepare the model as a C array
* prepare for compilation into `J9App.elf`

### Create project folder
Go to your working directory:
```bash
cd ~/project
```
Create the project structure:
```bash
mkdir -p mnist_e8_project/{build,linker,bsp,executorch/include,executorch/lib}
```

Now check:
```bash
tree mnist_e8_project
```

If `tree` is missing:
```bash
sudo apt install tree
```

Expected structure:
```bash
mnist_e8_project/
├── build
├── bsp
├── executorch
│   ├── include
│   └── lib
└── linker
```

### Create CMakeLists.txt
```bash
cd mnist_e8_project
```

Open nano editor:
```bash
nano CMakeLists.txt
```

Paste this:
```bash
cmake_minimum_required(VERSION 3.20)

project(J9App C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

#CPU compile flags
set(MCPU_FLAGS
    -mcpu=cortex-m85
    -mthumb
    -O3
    -ffunction-sections
    -fdata-sections
)

#Application executable
add_executable(J9App.elf
    J9App.cpp
    model_data.cpp
    syscalls.c
)

#Include directories
target_include_directories(J9App.elf PRIVATE
    CMSIS-NN/Include
    tflite-micro
    ethos-u-core-driver
)

#Compiler flags
target_compile_options(J9App.elf PRIVATE
    ${MCPU_FLAGS}
)

#Linker flags
target_link_options(J9App.elf PRIVATE
    ${MCPU_FLAGS}
    -Wl,--gc-sections
)

#Size output
add_custom_command(TARGET J9App.elf POST_BUILD
    COMMAND arm-none-eabi-size J9App.elf
)
```
Save:
* CTRL+O
* Enter
* CTRL+X

### Create Toolchain File
Open nano editor:
```bash
nano arm-none-eabi-toolchain.cmake
```

Paste:
```bash
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(TOOLCHAIN_PREFIX arm-none-eabi-)

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}gcc)

set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}objcopy)
set(CMAKE_SIZE ${TOOLCHAIN_PREFIX}size)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
```
Save and exit.

### Create Minimal Bare-Metal Syscall Stub

Bare-metal embedded applications do not run on a desktop operating system. Therefore, minimal syscall stubs are required by the Arm GNU Toolchain runtime libraries.

Create the file:
```bash
nano syscalls.c
```

Add the following code:
```c
void _exit(int status) {
    while (1) {
    }
}
```

This prevents linker errors related to missing system call implementations during firmware generation.

### Configure the Build System

Run the following command from the project directory:

```bash
cmake -B build \
-G Ninja \
-DCMAKE_TOOLCHAIN_FILE=arm-none-eabi-toolchain.cmake
```

This step configures the cross-compilation environment and generates Ninja build files.

(Build the J9App.elf)

```bash
cmake --build build
```

This command compiles the firmware and generates:

```text
J9App.elf
```

The generated ELF file is the executable firmware image for the Arm Cortex-M85 target platform. At this stage, the generated ELF only verifies that the cross-compilation environment, linker configuration, and firmware build system are functioning correctly.

### Note:
At this stage, the build may fail with:
```text
undefined reference to 'main'
```

This is expected because the embedded application source file (`J9App.cpp`) has not yet been implemented. Currently, J9App.cpp contains only placeholder application code. The actual TensorFlow Lite Micro inference application will be developed in the next stage.

So till now the project structure should look like this:
```text
project/
│
├── build/
├── data/
│
├── output/
│   ├── MyMNISTModel_int8_vela.tflite
│   ├── summary_REPORT.csv
│   └── vela.log
│
├── CMSIS-NN/
├── CMSIS_5/
├── ethos-u-core-driver/
├── ml-embedded-evaluation-kit/
├── tflite-micro/
│
├── final_mnist.keras
├── final_mnist.tflite
├── MyMNISTModel_int8.tflite
├── model_data.cpp
│
├── J9App.cpp
├── syscalls.c
├── CMakeLists.txt
├── arm-none-eabi-toolchain.cmake
│
└── build/
    └── J9App.elf
```


## Building the J9App.cpp

### Understanding what J9App really does
Internally the flow is:
```bash
Model (.tflite)
      ↓
TFLM Interpreter
      ↓
Tensor Arena Memory
      ↓
Input Tensor
      ↓
Invoke()
      ↓
Output Tensor
      ↓
Prediction
```



