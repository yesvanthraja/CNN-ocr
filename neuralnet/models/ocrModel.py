from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Activation, Dense, Flatten, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow import Tensor
from typing import Tuple, List

class OCRModel:
    @staticmethod
    def nn(data: Tensor, filters: int, stride: Tuple[int, int], chanDim: int, red: bool = False, reg: float = 0.0001, bnEps: float = 2e-5, bnMom: float = 0.9) -> Tensor:
        """
        Constructs a neural network.

        Args:
            data (Tensor): Input tensor.
            filters (int): Number of filters in the convolutional layers.
            stride (Tuple[int, int]): Stride of the convolutional layers.
            chanDim (int): Channel dimension index.
            red (bool): Whether to perform downsampling. Defaults to False.
            reg (float): Regularization parameter. Defaults to 0.0001.
            bnEps (float): Batch normalization epsilon. Defaults to 2e-5. to avoid zerodivision error
            bnMom (float): Batch normalization momentum. Defaults to 0.9.

        Returns:
            Tensor: Output tensor of the neural net module.
        """
        shortcut = data
        bnorm1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        actfunc1 = Activation("relu")(bnorm1)
        conv1 = Conv2D(int(filters * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(actfunc1)
        bnorm2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        actfunc2 = Activation("relu")(bnorm2)
        conv2 = Conv2D(int(filters * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(actfunc2)
        bnorm3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        actfunc3 = Activation("relu")(bnorm3)
        conv3 = Conv2D(filters, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(actfunc3)
        if red:
            shortcut = Conv2D(filters, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(actfunc1)
        x = add([conv3, shortcut])
        return x

    @staticmethod
    def build_model(width: int, height: int, depth: int, classes: int, stages: List[int], filters: List[int],reg: float = 0.0001, bnEps: float = 2e-5, bnMom: float = 0.9) -> Model:
        """
        Constructs a ResNet-based OCR model.

        Args:
            width (int): Width of the input images.
            height (int): Height of the input images.
            depth (int): Depth of the input images (number of channels).
            classes (int): Number of classes.
            stages (List[int]): List specifying the number of residual blocks in each stage.
            filters (List[int]): List specifying the number of filters in each stage.
            reg (float): Regularization parameter. Defaults to 0.0001.
            bnEps (float): Batch normalization epsilon. Defaults to 2e-5.
            bnMom (float): Batch normalization momentum. Defaults to 0.9.

        Returns:
            Model: Constructed OCR model.
        """
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)
        x = Conv2D(filters[0], (7, 7), strides=(2, 2), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        for i in range(0, len(stages)):
            stride = (2, 2) if i > 0 else (1, 1)
            x = OCRModel.nn(x, filters[i + 1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
            for j in range(0, stages[i] - 1):
                x = OCRModel.nn(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((3, 3))(x)
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        model = Model(inputs, x, name="ocr_extracter")
        return model