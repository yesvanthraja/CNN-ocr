# OCR Model Architecture 

The model architecture is designed to perform BatchNormalization over the image and leveraging convolutional neural networks to process and understand the contents of an image at various levels of abstraction.

## Note

Please check the `output_ocr_images` folder to view the OCR extracted images.

## Neural Network Module (`nn` method)

### Overall Model Architecture 
![GSOC-ocr-architechture-latest](https://github.com/yesvanthraja/CNN-ocr/assets/68420593/3fd3553e-aeff-455e-902f-dfb446f323b1)

### Model breakdown for explanation
<img width="713" alt="image" src="https://github.com/yesvanthraja/CNN-ocr/assets/68420593/89460833-c4a8-4687-8103-fafffcdd25dd">

- Input image is sent to the model 
- Preprocessing steps are performed(`Grayscle Conversion, Gaussian Blur, Canny Edge Detection, Thresholding, Resizing`)
- Image is passed to the `Convolutional layer` and `BatchNormalization` is performed.
- `ReLU` activation function is applied for avoiding Vanishing gradient.

     
<img width="719" alt="image" src="https://github.com/yesvanthraja/CNN-ocr/assets/68420593/db89b8bf-c587-4d5f-befd-561e4eaf688a"> 

- `Residual block` is introduced for feature reuse and reduce overfitting.
- `Global Average Pooling`(GAP) is introduced for dimentionality reduction and regularization.
- `Flatten` layer is used for converting the multidimensional output to single array of the output classes.

<img width="533" alt="image" src="https://github.com/yesvanthraja/CNN-ocr/assets/68420593/7251cbef-f154-4cff-b4e5-4f7d505d5377">

- `Dense Layer` is introduced for the classification of the classes.
- `Softmax` for converting output scores into probabilities for likelihood classification of each class.

                                               

### Parameters:
- `data`: Input tensor of shape `(batch_size, height, width, channels)` that represents the input images represents the output of the previous layer in the neural network.
- `filters`: The number of filters in the convolutional layers. Determines the depth of the feature maps. Capturing specific patterns or features of the input data.
- `stride`: The stride of the convolution operations. A stride of `(2, 2)` would reduce the spatial dimensions by half.
- `chanDim`: The channel dimension index, which adjusts based on the data format (`channels_first` or `channels_last`).
- `red` (optional): A boolean flag that, when `True`, adds a convolutional layer to the shortcut connection for dimensionality reduction.
- `reg` (optional): Regularization parameter to prevent overfitting by penalizing large weights.
- `bnEps` (optional): Small float added to the variance to avoid dividing by zero in batch normalization.
- `bnMom` (optional): Momentum for the moving average in batch normalization.

### Workflow:
1. **Initial Batch Normalization and Activation**: The input tensor is first normalized and then passed through a ReLU activation function.
2. **Convolution Block**: This consists of three convolutional layers, each followed by batch normalization and ReLU activation. The first and last convolutional layers use `(1, 1)` kernels for reducing and then expanding the number of filters, respectively, while the middle layer uses a `(3, 3)` kernel for spatial processing. The stride of the middle convolution layer controls downsampling.
3. **Dimensionality Reduction**: If `red` is `True`, the shortcut connection is passed through a `(1, 1)` convolutional layer to match the dimensions of the main path, facilitating element-wise addition.
4. **Skip Connection**: The output of the convolution block is added to the shortcut connection.

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


## Evaluation
### Word Error Rate (WER)
- Word Error Rate measures the difference between the recognized text and the ground truth text. It is calculated as the ratio of the total number of insertions, deletions, and substitutions required to convert the recognized text into the ground truth text, to the total number of words in the ground truth text.

The formula for calculating WER is as follows: Where:
- WER = (S + D + I) / N
- I = Number of insertions (words present in ground truth but not in recognized text)
- D = Number of deletions (words present in recognized text but not in ground truth)
- S = Number of substitutions (words in recognized text that are different from words in ground truth)
- N = Total number of words in ground truth

1. Input image -> sample_img1.jpeg
`original_text = ['G S O C']`
`extracted_text = ['G S O C']`
- S = 0 (substitution)
- D = 0 (deletion)
- I = 0 (insertion)
- N = Total number of words in the original text = 4
- Word error rate = 0+0+0/4 = 0 %

2. Input image -> sample_img2.jpeg
`original_text = ['H U M A N A I G S O C']`
`extracted_text = ['H U M A N AOI G S O C']`
- S = 1 (substitution)
- D = 0 (deletion)
- I = 0 (insertion)
- N = Total number of words in the original text = 5
- Word error rate = 1+0+0/11 ~ 9 %

So based on the Word Error Rate we can evaluate our model.

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
git clone https://github.com/yesvanthraja/CNN-ocr.git
```

2. Install the necessary Python packages using the provided requirements.txt file
```bash
pip install -r requirements.txt
```

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


