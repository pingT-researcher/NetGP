import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

# Define a class named DNNGP which inherits from the PyTorch's nn.Module class.
# This indicates that it is a module that can be used to build a neural network and might be used to implement functions related to Deep Neural Network Gaussian Process (DNNGP).
class DNNGP(nn.Module):
    def __init__(self, snp_len):
        """
        Initialization function for setting up the various layers and related parameters of the DNNGP model.

        :param snp_len: A parameter related to the length of the input data's features. It might be used later to determine the input dimension of fully connected layers and other operations based on the feature situation of the data.
                        Its specific meaning depends on the preprocessing of the data and the requirements of the overall model architecture for the input data.
        """
        super(DNNGP, self).__init__()
        # Define the first 2D convolutional layer. It has 24 input channels, 64 output channels, a kernel size of 3x3, a stride of 1, and a padding of 1.
        # This layer is used for feature extraction from the input data, capturing different local feature patterns through different convolutional kernels.
        self.conv1 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Define the first Dropout layer with a dropout probability of 0.1. During the training process, it randomly sets the outputs of some neurons to 0 with this probability.
        # This helps prevent overfitting and improves the generalization ability of the model.
        self.dropout1 = nn.Dropout(0.1)
        # Define the first Batch Normalization layer. It is used to normalize the data output from the convolutional layer, accelerating the convergence of the model and improving its stability.
        # Here, it performs batch normalization on the data with 64 output channels.
        self.batch_norm1 = nn.BatchNorm2d(64)
        # Define the second 2D convolutional layer. Its input channel number is 64 (the output channel number of the previous layer), and the output channel number is also 64. The kernel size and other parameters are similar to those of conv1.
        # It is used to further extract more complex features.
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Define the second Dropout layer with a dropout probability of 0.1. Its function is similar to that of the first Dropout layer, randomly dropping neuron outputs during the training stage.
        self.dropout2 = nn.Dropout(0.1)
        # Define the second Batch Normalization layer. It performs normalization on the data with 64 output channels to ensure the stability of the data distribution, which is beneficial for model training.
        self.batch_norm2 = nn.BatchNorm2d(64)
        # Define the third 2D convolutional layer. Both its input and output channel numbers are 64, continuing the feature extraction operation on the data.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Define a Flatten layer, which is used to flatten the multi-dimensional convolutional output results (usually in the form of feature maps) into a one-dimensional vector, so that it can be input into fully connected layers later.
        self.flatten = nn.Flatten()
        num = int(np.ceil(np.sqrt(np.ceil(snp_len / 24))))
        # Define the first fully connected layer. The input dimension is determined according to the feature map size output by the previous convolutional layers and related calculations (64 * num * num), and the output dimension is 64.
        # Its role is to integrate and map the features extracted by the convolutional layers to further extract high-level feature representations.
        self.fc1 = nn.Linear(64 * num * num, 64)
        # Define the second fully connected layer. Its input dimension is 64 (the output dimension of the previous layer), and the output dimension is 1. It is usually used for the final prediction task, outputting a scalar result.
        self.fc2 = nn.Linear(64, 1)
        # Define an instance of the ReLU activation function, which is used to introduce non-linearity into the model, enhancing the model's ability to fit complex data relationships.
        # The activation function will be applied to the outputs of multiple layers (such as after convolutional layers and fully connected layers) for non-linear transformation.
        self.relu = nn.ReLU()

    def forward(self, snps):
        """
        Forward propagation function that defines the flow and calculation order of data in the DNNGP model. That is, starting from the input data, it goes through the processing of each layer and finally obtains the output of the model.

        :param snps: The input data tensor. Its shape and dimensions should meet the input requirements of the first layer of the model (self.conv1).
                    The specific format depends on the preprocessing of the data and the overall application scenario, and usually contains the core feature information that the model needs to process.
        :return: Returns the output tensor after the entire DNNGP model has processed the input data. The shape is usually (batch_size, 1), where batch_size depends on the batch size of the input data.
                 The final output dimension of 1 corresponds to the prediction result of the model (such as a certain score, probability, etc., depending on the application scenario of the model).
        """
        # The following line of code is commented out. It might have been originally used to add a dimension to the input data (if the input data dimension did not meet the requirements of the convolutional layer).
        # For example, it could add the channels dimension to the shape like (batch_size, channels, height, width) to make it conform to the input specification of the 2D convolutional layer.
        # snps_in = torch.unsqueeze(snps, dim=1) 
        # Pass the input data snps into the first convolutional layer self.conv1 for the convolution operation, and then apply the ReLU activation function to the convolution result for non-linear transformation to extract the initial local features.
        x = self.relu(self.conv1(snps))
        # Apply the first Dropout layer to the result after convolution and activation, randomly dropping the outputs of some neurons to prevent overfitting.
        x = self.dropout1(x)
        # Apply the first Batch Normalization layer to the data after Dropout processing to perform normalization operation and make the data distribution more stable.
        x = self.batch_norm1(x)
        # Pass the data that has gone through the above processing into the second convolutional layer self.conv2 for further convolution operation, and then apply the ReLU activation function to obtain more complex features.
        x = self.relu(self.conv2(x))
        # Apply the second Dropout layer to the data output by the second convolutional layer to continue preventing overfitting.
        x = self.dropout2(x)
        # Apply the ReLU activation function again to the data after Dropout processing to further enhance the non-linearity and extract more effective features (here, the activation function is applied before the third convolutional layer).
        x = self.relu(self.conv3(x))
        # Use the Flatten layer to flatten the multi-dimensional feature map data processed by the convolutional layers into a one-dimensional vector so that it can be input into the fully connected layers.
        x = self.flatten(x)
        # Pass the flattened one-dimensional data into the first fully connected layer self.fc1 for feature integration and mapping, and then apply the ReLU activation function for non-linear transformation.
        x = self.relu(self.fc1(x))
        # Finally, pass the data processed by the first fully connected layer into the second fully connected layer self.fc2 for the final feature mapping to obtain the output prediction result of the model,
        # and apply the ReLU activation function again (whether it is appropriate specifically depends on the specific task and model design requirements).
        x = self.relu(self.fc2(x))

        return x