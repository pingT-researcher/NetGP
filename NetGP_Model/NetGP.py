import torch.nn as nn
import torch
import numpy as np
import torch
import torch.nn as nn
from GCN import GraphConvolutionLayer
from torch import nn
import torch
import torch.nn.functional as F
from SparseConnectedModule import SparseConnectedModule


# Define a neural network model class named NetGS that inherits from nn.Module. It is used to build a neural network
# with specific structure and functions to process relevant input data and generate outputs.
class NetGP(nn.Module):
    def __init__(self, genes_mask, gene_re_mask):
        """
        Initialization function for creating an instance of the NetGS model and initializing each layer and related
        parameters in the model.

        Parameters: - genes_mask: Gene-related mask (usually in the form of a tensor, whose specific structure and
        meaning depend on the actual application scenario. It might be used to define sparse connection operations,
        etc.). Here, it is moved to the specified GPU device (cuda:0) for subsequent computations on the GPU (if a
        GPU is available). - gene_re_mask: Also a gene-related mask, which is used in the GraphConvolutionLayer to
        determine the structure, connection relationships, etc. of the graph. Its length will be used to define the
        input and output dimensions of the graph convolution layer and some subsequent linear layers.
        """
        super(NetGP, self).__init__()
        self.genes_mask = genes_mask.to('cuda:0')
        # Create an instance of the SparseConnectedModule and pass in genes_mask as an argument. This module might
        # implement sparse connection operations based on the mask and is used to process the input data.
        self.snp_lin = SparseConnectedModule(self.genes_mask)

        # Create the first instance of the GraphConvolutionLayer for performing graph convolution operations on
        # gene-related data. The input feature dimension and output feature dimension are both set to the length of
        # gene_re_mask. The third parameter passes in gene_re_mask to define the structure of the graph,
        # etc. The last parameter '3' might indicate a specific graph convolution calculation mode or configuration (
        # depending on the implementation).
        self.gene_Gnn1 = GraphConvolutionLayer(len(gene_re_mask), len(gene_re_mask), gene_re_mask, '3')
        # Create the second instance of the GraphConvolutionLayer, which is similar to the first one but might have a
        # different calculation mode (distinguished by the last parameter '1') and is used to further process the
        # features after graph convolution.
        self.gene_Gnn2 = GraphConvolutionLayer(len(gene_re_mask), len(gene_re_mask), gene_re_mask, '1')
        m = int(len(gene_re_mask))
        # Create the first linear layer that maps features with an input dimension of m (i.e., the dimension
        # corresponding to the length of gene_re_mask) to a feature space with a dimension of 32. It is used to
        # transform the features processed by the sparse connection module.
        self.pathway_lin1 = nn.Linear(m, 32)
        # Create the second linear layer that also maps features with an input dimension of m to a feature space with
        # a dimension of 32. However, it is used to transform the features processed by the graph convolution layer.
        self.pathway_lin2 = nn.Linear(m, 32)

        # Create a one-dimensional batch normalization layer for normalizing the concatenated features with a
        # dimension of 64. This helps improve the stability, convergence speed, and other performance aspects of the
        # model training.
        self.BN = nn.BatchNorm1d(64)
        # Create a linear layer that maps features with a dimension of 64 to an output space with a dimension of 1.
        # It is usually used to generate the final prediction result, etc.
        self.lin = nn.Linear(64, 1)
        # Define an instance of the ReLU activation function, which is used to introduce non-linearity into the
        # network and enhance the model's ability to learn complex data relationships.
        self.relu = nn.ReLU()

        # Define an instance of the Sigmoid activation function, which might be used later to transform certain
        # parameters to an appropriate value range (such as the interval [0, 1]) to implement functions like weight
        # allocation, etc.
        self.sigmal = nn.Sigmoid()
        # Create a learnable parameter tensor containing two elements, initialized as [0.8, 0.2], and set it to
        # require gradient calculation. It can be automatically updated during the training process and might be used
        # to fuse features from different paths, acting like weight adjustment.
        self.param = nn.Parameter(torch.FloatTensor([0.8, 0.2]), requires_grad=True)

    def forward(self, datas, transcribe):
        """
        Forward propagation function that defines the calculation process when input data passes through the entire NetGS model.
        It processes the data step by step according to the established network structure and finally generates the output.

        Parameters:
        - datas: Input data (the specific format and meaning depend on the actual application scenario. In the current model, it might be related to
                 Single Nucleotide Polymorphisms (SNP), etc., and will first be processed by the sparse connection module).
        - transcribe: Transcription-related data (similarly, its specific content is related to the application scenario. In this model, it will be used
                      as the input data for the graph convolution layer for processing).

        Returns:
        - The final output result after processing by all layers of the entire model. The output value after passing through the ReLU activation,
          usually representing the prediction result or processed feature representation of the model for the input data.
        """
        # Pass the input datas through the SparseConnectedModule for processing, applying the sparse connection
        # mechanism for feature transformation and obtaining the processed feature representation.
        snp_out = self.snp_lin(datas)
        # Apply the ReLU activation function to the features processed by the sparse connection module to introduce
        # non-linearity, enabling the model to learn more complex data patterns.
        snp_out = self.relu(snp_out)

        # Pass the features processed by the sparse connection module and after the ReLU activation into the
        # pathway_lin1 linear layer for dimension transformation and feature mapping, mapping them to a feature space
        # with a dimension of 32.
        pathway_out = self.pathway_lin1(snp_out)
        pathway_out1 = self.relu(pathway_out)

        # Pass the transcription-related transcribe data into the first graph convolution layer gene_Gnn1 for graph
        # convolution operations, obtaining the feature representation after graph convolution transformation.
        gnn_trans = self.gene_Gnn1(transcribe)
        # Apply the ReLU activation function to the features output by the first graph convolution layer to introduce
        # non-linearity and enhance the model's expressive ability.
        gnn = self.relu(gnn_trans)
        # Pass the features after the first graph convolution layer and the ReLU activation into the second graph
        # convolution layer gene_Gnn2 for further graph convolution processing.
        gnn = self.gene_Gnn2(gnn)
        # Apply the ReLU activation function again to activate the output of the second graph convolution layer and
        # maintain the non-linearity.
        gnn = self.relu(gnn)

        # Pass the features processed by the graph convolution layer into the pathway_lin2 linear layer for dimension
        # transformation and feature mapping, mapping them to a feature space with a dimension of 32.
        pathway_out = self.pathway_lin2(gnn)
        pathway_out2 = self.relu(pathway_out)

        # Apply the Sigmoid activation function to the learnable parameter self.param to transform its element values
        # to the interval [0, 1], so that they can be used as weights for feature fusion operations later.
        param = self.sigmal(self.param)

        # According to the weights param processed by the Sigmoid function, perform weighted concatenation on the
        # 32-dimensional features obtained from the two different paths, pathway_out1 and pathway_out2,
        # along dimension 1 (usually the feature dimension), obtaining a fused feature representation with a
        # dimension of 64.
        pathway_out = torch.cat((param[0] * pathway_out1, param[1] * pathway_out2), dim=1)

        # Apply the batch normalization layer to the concatenated 64-dimensional features for normalization
        # operations, which helps improve the stability and convergence effect of the model training.
        pathway_out = self.BN(pathway_out)
        # Pass the features after batch normalization into the final linear layer lin to map them to an output space
        # with a dimension of 1 and obtain the final output result.
        out = self.lin(pathway_out)
        return self.relu(out)
