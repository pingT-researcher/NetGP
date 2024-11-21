import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the graph convolution layer class, which inherits from nn.Module
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, n):
        """
        Initialization function for the graph convolution layer.

        Parameters:
        - in_features: The dimension size of the input features.
        - out_features: The dimension size of the output features.
        - adj: The adjacency matrix, representing the connection relationship of the graph.
        - n: An identifier used to select different calculation methods, which can take values like '1', '2', or other cases.
        """
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Define the bias term as a learnable parameter, initialized as a random value and set to require gradient
        # calculation, and then initialize it to 0.
        self.bias = nn.Parameter(torch.rand(out_features), requires_grad=True)
        nn.init.zeros_(self.bias)
        # Define the ReLU activation function.
        self.relu = nn.ReLU()
        # Convert elements in the adjacency matrix that are greater than 1 to floating-point number 1, and those less
        # than or equal to 1 to floating-point number 0, obtaining a new adjacency matrix adj1.
        adj1 = (adj > 1).float()
        # Convert elements in the adjacency matrix that are greater than or equal to 1 to floating-point number 1,
        # and those less than 1 to floating-point number 0, obtaining a new adjacency matrix adj2.
        adj2 = (adj >= 1).float()
        if n == '1':
            # Select the corresponding degree matrix calculation method according to the condition. Here,
            # if n is '1', call sqrt_degree_matrix to calculate the degree matrix.
            self.degree_matrix = self.sqrt_degree_matrix(adj1)
        elif n == '2':
            self.degree_matrix = self.sqrt_degree_matrix(adj2)
        else:
            self.degree_matrix = self.sqrt_degree_matrix(adj)
        # Use the degree matrix as a learnable weight parameter and set it to require gradient calculation.
        self.weight = nn.Parameter(self.degree_matrix, requires_grad=True)
        # Convert elements in the degree matrix whose absolute value is greater than 0 to floating-point number 1,
        # and then move it to the specified GPU device (cuda:0).
        self.degree_matrix = (torch.abs(self.degree_matrix) > 0).float().to('cuda:0')

    def sqrt_degree_matrix(self, adj):
        """
        Function to calculate the square root of the degree matrix.

        Parameters:
        - adj: The input adjacency matrix.

        Returns:
        - The degree matrix after specific calculations (already moved to the cuda:0 device).
        """
        # Calculate the sum of elements in each row of the adjacency matrix to obtain the degree (out-degree or
        # in-degree, depending on the definition of the adjacency matrix) of each node, forming the degree matrix (in
        # the form of a one-dimensional tensor).
        degree_matrix = torch.sum(adj, dim=1)
        # Perform square root operation on the degree matrix to construct a diagonal matrix, where the elements on
        # the diagonal are the reciprocal square roots of the corresponding degrees. Here, if the degree is 0,
        # infinity will occur, which will be handled later.
        sqrt_degree_matrix = torch.diag(1 / torch.sqrt(degree_matrix))
        # Set the elements with infinity to 0 to avoid problems in subsequent calculations.
        sqrt_degree_matrix[sqrt_degree_matrix == float('inf')] = 0
        # Perform matrix multiplication operation, which is equivalent to part of the normalization operation on the
        # adjacency matrix based on the degree matrix.
        support = torch.mm(sqrt_degree_matrix, adj)
        # Continue with matrix multiplication operation to further complete the normalization and add twice the
        # identity matrix (which may be used to prevent certain situations, such as overfitting, etc.).
        sqrt_degree_matrix = torch.mm(support, sqrt_degree_matrix) + (2 * torch.eye(self.in_features))
        return sqrt_degree_matrix.to('cuda:0')

    def forward(self, x):
        """
        Forward propagation function, which defines the calculation process of data passing through this graph convolution layer.

        Parameters:
        - x: The input feature tensor.

        Returns:
        - The output feature tensor after the calculation of the graph convolution layer.
        """
        # Perform element-wise multiplication operation on matrices, multiplying the weight and the degree matrix (
        # the specific calculation logic here needs to be understood in depth in combination with the actual situation).
        output = torch.mul(self.weight, self.degree_matrix)
        # Perform a linear transformation, combining the input feature x linearly with the previously calculated
        # output and adding the bias term self.bias.
        output = F.linear(x, output, self.bias)
        return output
