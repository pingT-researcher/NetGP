from torch import nn
import torch
import torch.nn.functional as F


# Define a class named SparseConnectedModule that inherits from nn.Module.
# This class is used to create a module with sparse connection characteristics.
class SparseConnectedModule(nn.Module):
    def __init__(self, custom_mask):
        """
        Initialize the SparseConnectedModule.

        Args: custom_mask: A custom mask (usually in the form of a tensor) which is used to define the sparsity of
        connections. It determines which connections are valid and which are invalid. This mask will be treated as a
        learnable weight parameter.
        """
        super(SparseConnectedModule, self).__init__()
        # Set the custom_mask as a learnable weight parameter.
        # This means its values can be updated during the training process to optimize the model's performance.
        self.weight = nn.Parameter(custom_mask, requires_grad=True)
        # Initialize the bias as a learnable parameter with random values.
        # Its length is the same as that of the custom_mask and it will be used in the subsequent linear calculation.
        self.bias = nn.Parameter(torch.rand(len(custom_mask)), requires_grad=True)

    def forward(self, x):
        """
        Define the forward propagation process of the module.

        Args: x: The input data tensor. Its shape should be compatible with the module's weight and other parameters
        for the subsequent linear transformation.

        Returns: The output tensor after the linear transformation, which is based on the defined weight (
        self.weight) and bias (self.bias).
        """
        # Perform a linear transformation on the input x using the PyTorch's F.linear function.
        # The function calculates the output based on the given weight (self.weight) and bias (self.bias).
        x = F.linear(x, self.weight, self.bias)
        return x
