# OCR Model Architecture Explanation

This Optical Character Recognition (OCR) model is built using a ResNet-based architecture, tailored for recognizing text in images. The model architecture is designed to leverage deep convolutional neural networks to process and understand the contents of an image at various levels of abstraction. Below is a line-by-line explanation of the model's components and their parameters:

## Neural Network Module (`nn` method)

### Parameters:
- `data`: Input tensor of shape `(batch_size, height, width, channels)` that represents the input images.
- `filters`: The number of filters in the convolutional layers. Determines the depth of the feature maps.
- `stride`: The stride of the convolution operations. A stride of `(2, 2)` would reduce the spatial dimensions by half.
- `chanDim`: The channel dimension index, which adjusts based on the data format (`channels_first` or `channels_last`).
- `red` (optional): A boolean flag that, when `True`, adds a convolutional layer to the shortcut connection for dimensionality reduction.
- `reg` (optional): Regularization parameter to prevent overfitting by penalizing large weights.
- `bnEps` (optional): Small float added to the variance to avoid dividing by zero in batch normalization.
- `bnMom` (optional): Momentum for the moving average in batch normalization.

### Workflow:
1. **Shortcut Connection Setup**: A shortcut connection is set to the input data. This aids in mitigating the vanishing gradient problem by allowing gradients to flow through the network more effectively.
2. **Initial Batch Normalization and Activation**: The input tensor is first normalized and then passed through a ReLU activation function.
3. **Convolution Block**: This consists of three convolutional layers, each followed by batch normalization and ReLU activation. The first and last convolutional layers use `(1, 1)` kernels for reducing and then expanding the number of filters, respectively, while the middle layer uses a `(3, 3)` kernel for spatial processing. The stride of the middle convolution layer controls downsampling.
4. **Dimensionality Reduction**: If `red` is `True`, the shortcut connection is passed through a `(1, 1)` convolutional layer to match the dimensions of the main path, facilitating element-wise addition.
5. **Skip Connection**: The output of the convolution block is added to the shortcut connection.

## Model Building (`build_model` method)

### Parameters:
- `width`, `height`, `depth`: Dimensions of the input images.
- `classes`: The number of classes for classification.
- `stages`: Specifies the number of residual blocks in each stage of the model.
- `filters`: Specifies the number of filters for each stage.
- Additional parameters for regularization and batch normalization as described above.

### Workflow:
1. **Input and Initial Convolution**: The model begins with a standard input layer followed by an initial convolutional layer with a `(7, 7)` kernel and a stride of `(2, 2)`, immediately followed by batch normalization and ReLU activation.
2. **Residual Blocks Construction**: For each stage specified in `stages`, a series of residual blocks is constructed using the `nn` method. The first block of each stage may perform downsampling depending on the stride.
3. **Final Layers**: After all stages, the model applies a final batch normalization and ReLU activation, followed by average pooling, flattening, and a dense layer for classification.
4. **Output Activation**: A softmax activation function is used to output the probabilities for each class.

The model is designed to be flexible, allowing customization of depth, filter sizes, and regularization parameters to adapt to different OCR tasks. The use of residual blocks helps in training deeper networks by addressing the vanishing gradient problem.

## Installation

To utilize this OCR model, follow the steps below for setting up your environment and installing the required dependencies.

### Requirements

- Python 3.6 or later
- TensorFlow 2.x
- Keras
- NumPy

### Setup

1. Clone the repository to your local machine:

```bash
git clone <https://github.com/yesvanthraja/CNN-ocr.git>
```

2. Install the necessary Python packages using the provided requirements.txt file
`pip install -r requirements.txt`

3. Training the Model
To train the OCR model, you'll need a dataset of images with corresponding text annotations. This project uses a CSV file, handwritten_data.csv, which contains paths to images and their annotations.

    Run the following command to start the training process:
```bash
python train.py --az handwritten_data.csv --model ocrmodel/ocr.keras
```
- Arguments:
- --az: Path to the annotations CSV file.
- --model: Path where the trained model should be saved.
This script will train the OCR model using the specified dataset and save the trained model to the provided path.

4. Testing the Model
After training, you can test the OCR model on new images to recognize handwritten text.

    To test the model, use the following command:
```bash
python test.py --model ocrnew.h5 --image images/sample_img1.jpeg
```

- Arguments:
- --model: Path to the trained OCR model file.
- --image: Path to the image file on which OCR detection should be performed.
This command will output the recognized text from the provided image using the trained OCR model.


This README.md format offers a comprehensive guide to using your OCR model, including how to set up the environment, train the model on a dataset, and test it on new images. It should help users understand and effectively utilize the OCR model for their handwritten text recognition tasks.


