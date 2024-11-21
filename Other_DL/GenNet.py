import torch.nn as nn
import torch
import numpy as np
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

# Define a class named SparseConnectedModule, which inherits from PyTorch's nn.Module class.
# It is used to create a neural network module with a specific sparse connection structure.
class SparseConnectedModule(nn.Module):
    def __init__(self, input_size, output_size, custom_mask):
        """
        Initialization function used to set various parameters and attributes of the module.

        :param input_size: The number of features of the input data, which is equivalent to the size of the last dimension of the input tensor.
        :param output_size: The number of features of the output data, also representing the size of the last dimension of the output tensor of this module.
        :param custom_mask: A custom mask tensor used to define the sparse connection structure of the weight matrix. Its shape should match the transpose of the weight matrix.
                           Usually, it is a Boolean or 0/1 tensor, used to control which weights are involved in the calculation (weights corresponding to positions with a value of 1 participate in the calculation, while those with a value of 0 do not).
        """
        super(SparseConnectedModule, self).__init__()
        # Transpose the incoming custom mask tensor and move it to the specified GPU device (cuda:0) for efficient computation on the GPU later.
        self.mask = custom_mask.T.to('cuda:0')
        # Define the learnable weight parameter. Initialize a random tensor with a shape of (output_size, input_size) using torch.rand, and set requires_grad to True, indicating that this weight will be updated according to the gradient during the training process.
        self.weight = nn.Parameter(torch.rand(output_size, input_size), requires_grad=True) 
        # Define the learnable bias parameter. Initialize a random tensor with a length of output_size using torch.rand, and also set requires_grad to True so that it can be updated during training and is used to adjust the offset of the result of the linear transformation.
        self.bias = nn.Parameter(torch.rand(output_size), requires_grad=True)

    def forward(self, x):
        """
        Forward propagation function that defines how data flows and is calculated within this module. That is, given the input data x, it specifies how to obtain the output.

        :param x: The input data tensor, whose shape should match the defined input size (input_size). It is usually in the form of (batch_size, input_size), where batch_size represents the batch size, meaning the number of data samples input at one time.
        :return: Returns the output tensor after linear transformation and adding the bias. The shape is (batch_size, output_size), where batch_size remains unchanged, and output_size is the defined number of output features.
        """
        # Multiply the weight tensor and the mask tensor element by element to screen the weights according to the sparse connection structure defined by the mask and obtain the weight matrix that actually participates in the calculation.
        mask_weight = torch.mul(self.weight, self.mask)
        # Use the F.linear function to perform a linear transformation. Multiply the input data x by the weight matrix mask_weight after being processed by the mask, and then add the bias self.bias to obtain the result of the linear transformation. This function internally handles the dimension situation of batch data automatically and calculates according to the rules of matrix multiplication.
        x = F.linear(x, mask_weight, self.bias)
        return x

# Define a class named GenNet, which also inherits from the nn.Module class. It is likely used to build a neural network model based on gene-related information.
class GenNet(nn.Module):
    def __init__(self, genes_mask):
        """
        Initialization function used to construct the structure of the GenNet model, including specific layers and modules.

        :param genes_mask: The gene mask tensor. Its structure and meaning are similar to the custom_mask in the SparseConnectedModule.
                           It may be used to define the sparse connection structure related to genes, for example, to control which gene features participate in the calculation of the model.
        """
        super(GenNet, self).__init__()
        # Create an instance of the SparseConnectedModule and pass in the gene mask tensor genes_mask. It is used to handle the initial sparse linear transformation of the input data.
        # The input size is the number of rows of the gene mask tensor (related to the dimension of the number of genes), and the output size is the number of columns of the gene mask tensor, which may correspond to a certain transformed feature dimension.
        self.snp_lin = SparseConnectedModule(len(genes_mask), len(genes_mask[0]), genes_mask)
        # Create a common linear layer, which is used to further map the features processed by the previous sparse connection module to an output space with a dimension of 1.
        # The input size is the feature dimension after the previous processing (len(genes_mask[0])), and the output size is 1, which may be used for final prediction or feature integration purposes.
        self.pathway_layer = nn.Linear(len(genes_mask[0]), 1)
        # Create an instance of the ReLU activation function, which is used to introduce non-linearity into the model, activate the output of the intermediate layer, and increase the expressive ability of the model.
        self.relu = nn.ReLU()

    def forward(self, datas):
        """
        Forward propagation function that defines the flow and calculation order of data in the entire GenNet model. Starting from the input data, it goes through the processing of each layer and finally obtains the output of the model.

        :param datas: The input data tensor, whose shape should match the expected input shape of the first layer of the model (self.snp_lin).
                      It usually contains information related to gene features, etc., depending on the structure and preprocessing of the dataset.
        :return: Returns the output tensor after the entire GenNet model has processed the input data. The shape is usually (batch_size, 1), where batch_size depends on the batch size of the input data.
                 The final output dimension of 1 may correspond to the prediction result of the model (such as a certain score, probability, etc., depending on the application scenario of the model).
        """
        # Pass the input data datas into the sparse connection module self.snp_lin for processing, and obtain the output result after the first linear transformation and sparse connection screening.
        snp_out = self.snp_lin(datas)
        # Apply the ReLU activation function to the output result of the sparse connection module to introduce non-linear transformation, enabling the model to learn more complex patterns.
        snp_out = self.relu(snp_out)
        # Pass the activated result into the common linear layer self.pathway_layer for further linear transformation, mapping the features to an output space with a dimension of 1.
        out = self.pathway_layer(snp_out)
        # Apply the ReLU activation function to the output result of the linear layer again. This may be done to ensure that the output result meets certain specific value ranges or property requirements, depending on the design and application scenario of the model. Finally, obtain the output of the model and return it.
        out = self.relu(out)
        return out