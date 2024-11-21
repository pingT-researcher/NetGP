import torch.nn as nn
import torch
import numpy as np
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import torch.nn.init as nninit
import torch.nn.functional as F
from GCN import GraphConvolutionLayer
from torch import nn
import torch
import torch.nn.functional as F
from SparseConnectedModule import SparseConnectedModule


# Define a neural network model class named NetGS, which inherits from nn.Module and is used to build a neural
# network with a specific structure.
class NetGP(nn.Module):
    def __init__(self, genes_mask, gene_re_mask):
        """
        Initialization function, used to create an instance of the NetGS model and set the parameters and structure
        of each layer.

        Parameters: - genes_mask: Gene-related mask (the specific meaning needs to be determined according to the
        actual application scenario, and it may be used for filtering, identification, etc.). - gene_re_mask: Another
        gene-related mask, which is used in the graph convolution layer to define the graph structure and other
        related operations in the subsequent code (similarly, the specific role depends on the specific application
        scenario).
        """
        super(NetGP, self).__init__()
        self.gene_Gnn1 = GraphConvolutionLayer(len(gene_re_mask), len(gene_re_mask), gene_re_mask, '3')
        self.gene_Gnn2 = GraphConvolutionLayer(len(gene_re_mask), len(gene_re_mask), gene_re_mask, '1')
        m = int(len(gene_re_mask))
        self.pathway_lin = nn.Linear(m, 32)
        self.BN_P = nn.BatchNorm1d(32)
        self.lin = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, datas, transcribe):
        """
        Forward propagation function, which defines the calculation process when data passes through the entire NetGS model.

        Parameters:
        - datas: SNP data
        - transcribe: Transcription data

        Returns:
        - The result output after the entire model's calculation, the final output value after passing through the ReLU activation.
        """
        gnn_trans = self.gene_Gnn1(transcribe)
        gnn = self.relu(gnn_trans)
        gnn = self.gene_Gnn2(gnn)
        gnn = self.relu(gnn)
        pathway_out = self.pathway_lin(gnn)
        pathway_out = self.relu(pathway_out)
        # Apply the batch normalization layer to the features with a dimension of 32 for normalization operations,
        # which helps improve the stability and effectiveness of model training.
        pathway_out = self.BN_P(pathway_out)
        out = self.lin(pathway_out)
        return self.relu(out)
