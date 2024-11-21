import torch
import torch.nn as nn
import torch.nn.init as init

# Define a class named DeepGS which inherits from the PyTorch's nn.Module class. This indicates that it is a neural network model that can be used to implement specific functions (perhaps related to genes depending on the application scenario).
class DeepGS(nn.Module):
    def __init__(self, snp_len):
        """
        Initialization function for constructing the network structure of the DeepGS model, defining each layer and related parameters.

        :param snp_len: A parameter related to the length of the input data's features. It is used to determine the input and output dimensions of some layers in the network based on the dimension information of the input data.
                        For example, the calculation of the input dimension of the subsequent fully connected layers depends on this parameter. Its specific meaning depends on the data preprocessing method and the requirements of the overall model architecture.
        """
        super(DeepGS, self).__init__()
        # Define a 1D convolutional layer (Conv1d). The number of input channels is 1, indicating single-channel features of the input data. The number of output channels is 8, meaning that 8 different feature channels will be generated after the convolution operation.
        # The kernel size is 18 and the stride is 1. This layer is used for feature extraction from the input data, capturing local feature patterns along the one-dimensional direction through different convolutional kernels.
        self.CNN = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=18, stride=1)
        # Define a 1D max pooling layer (MaxPool1d). The pooling kernel size is 4 and the stride is 4. Its role is to downsample the data output from the convolutional layer, reducing the data dimension while retaining important feature information.
        # This helps to extract more representative features, reduce the computational load, and to some extent prevent overfitting.
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        # Calculate the length of the data in the one-dimensional direction after convolution and pooling based on the input data length (snp_len), the kernel size, and the parameters of the pooling operation.
        # This calculation result will be used to determine the input dimension of the subsequent fully connected layers to ensure the matching of data dimensions.
        n = (int)((snp_len - 18 + 1) / 4)
        # Define the first fully connected layer. The input dimension is the dimension of the data after pooling (n * 8, where n is the length calculated previously and 8 is the number of channels output by the convolutional layer). The output dimension is 32.
        # Its purpose is to integrate and map the features extracted by convolution and pooling to further extract higher-level feature representations to adapt to subsequent processing and prediction tasks.
        self.linear1 = nn.Linear(n * 8, 32)
        # Define the second fully connected layer. The input dimension is 32 (the output dimension of the previous fully connected layer), and the output dimension is 1. It may be used to map the features to the final prediction space and output a scalar result,
        # such as predicting a specific value or probability, depending on the application scenario of the model.
        self.linear2 = nn.Linear(32, 1)
        # Define the third fully connected layer. Both the input and output dimensions are 1. Its role may be to further adjust or adapt the previous result. The specific function also depends on the specific requirements of the model design.
        self.linear3 = nn.Linear(1, 1)
        # Define a Dropout layer with a dropout probability of 0.2. During the training process, the outputs of some neurons will be randomly set to 0 according to this probability.
        # This is done to prevent the model from overfitting and improve its generalization ability. Here, it is mainly applied to the data after the pooling layer.
        self.dropout2 = nn.Dropout(0.2)
        # Define another Dropout layer with a dropout probability of 0.1. It is used to randomly drop the outputs of some neurons after the first fully connected layer to avoid overfitting.
        self.dropout1 = nn.Dropout(0.1)
        # Define another Dropout layer with a dropout probability of 0.05. It is used to perform a similar overfitting prevention operation after the second fully connected layer. Different dropout probability settings may be based on different requirements for preventing overfitting in different layers.
        self.dropout05 = nn.Dropout(0.05)
        # Define an instance of the ReLU activation function, which is used to introduce non-linearity into the model, enabling it to learn more complex data relationships.
        # The activation function will be applied to the outputs of multiple layers (such as after the convolutional layer, between fully connected layers, etc.) for non-linear transformation.
        self.relu = nn.ReLU()
        # Define an instance of the Sigmoid activation function, which is usually used to map the output to a probability value range between 0 and 1. It may be used in the final output layer or in some specific places where the result needs to be converted into a probability form,
        # depending on the specific application scenario and prediction goal of the model.
        self.sigmoid = nn.Sigmoid()
        # Define a Flatten layer, which is used to flatten multi-dimensional data (for example, the multi-dimensional feature representation after convolution and pooling) into a one-dimensional vector,
        # so that it can be correctly input into the fully connected layers for processing and ensure the matching of data dimensions.
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Forward propagation function that defines the flow and calculation order of data in the DeepGS model. Starting from the input data, it goes through the processing of each layer and finally obtains the output of the model.

        :param x: The input data tensor. Its shape and dimensions should meet the input requirements of the first layer of the model (self.CNN).
                  It usually contains the core feature information that the model needs to process. The specific format depends on the data preprocessing and the application scenario of the model.
        :return: Returns the output tensor after the entire DeepGS model has processed the input data. The shape is usually (batch_size, 1), where batch_size depends on the batch size of the input data.
                 The final output dimension of 1 corresponds to the prediction result of the model (such as a certain score, probability, etc., depending on the application scenario of the model).
        """
        # Add a dimension to the input data x at dimension 1, transforming its shape from (batch_size, feature_size) to (batch_size, 1, feature_size).
        # This makes it meet the dimension requirements of the 1D convolutional layer (Conv1d), which requires a representation of the channel dimension.
        snp_out = torch.unsqueeze(x, dim=1)
        # Pass the input data with adjusted dimensions, snp_out, into the 1D convolutional layer self.CNN for the convolution operation to extract local features of the data.
        cnn_out = self.CNN(snp_out)

        # Apply the ReLU activation function to the output of the convolutional layer for non-linear transformation, enhancing the model's ability to fit complex data relationships and obtaining the activated convolutional output result.
        cnn_relu = self.relu(cnn_out)
        # Pass the activated convolutional result into the max pooling layer self.pool for downsampling operation to reduce the data dimension and extract more representative features.
        pool_out = self.pool(cnn_relu)
        # Apply the Dropout operation to the data output by the pooling layer. Randomly set the outputs of some neurons to 0 according to the set dropout probability of 0.2 to prevent overfitting and obtain the pooling output result after Dropout processing.
        pool_out_dp = self.dropout2(pool_out)
        # Use the Flatten layer self.flatten to flatten the pooling output result after Dropout processing (which may be a multi-dimensional feature representation) into a one-dimensional vector so that it can be input into the fully connected layers.
        flatten = self.flatten(pool_out_dp)

        # Pass the flattened one-dimensional data into the first fully connected layer self.linear1 for feature integration and mapping to obtain the output result of the fully connected layer.
        lin_1 = self.linear1(flatten)
        # First, apply the Dropout operation to the output of the first fully connected layer, randomly dropping the outputs of some neurons according to the probability of 0.1, and then apply the ReLU activation function for non-linear transformation.
        # This further enhances the non-linearity of the model and prevents overfitting at the same time, obtaining the processed output result of the fully connected layer.
        lin_1_relu_dp = self.relu(self.dropout1(lin_1))

        # Pass the processed result into the second fully connected layer self.linear2 for further feature mapping to further transform the features into an appropriate space for subsequent processing or final output.
        lin_2 = self.linear2(lin_1_relu_dp)

        # Apply the Dropout operation to the output of the second fully connected layer, randomly dropping the outputs of some neurons according to the probability of 0.05, and then pass it into the third fully connected layer self.linear3 for final feature mapping.
        # Obtain the final output result of the model. The specific meaning of this result (such as whether it is a probability value or other predicted values) depends on the application scenario of the model.
        lin_3 = self.linear3(self.dropout05(lin_2))
        return lin_3