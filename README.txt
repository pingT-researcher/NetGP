###NetGP
#This project provides various models and data processing tools, including the NetGP model, deep learning and machine learning comparison experiments, as well as data preprocessing utilities. The code is organized into four main modules: Deal_Datas, NetGP_Model, Other_DL, and Other_ML. Each module implements specific functions for data processing, model training, and evaluation.

Environment Setup
This project runs in the following environment:
Python: 3.6
PyTorch: 1.7.0
CUDA: 11.0
Installing Dependencies
Create a new Conda environment:
conda create --name myenv python=3.6

Activate the environment:
conda activate myenv

Install the required dependencies: Make sure you have a requirements.txt file listing all the necessary dependencies. Install them by running:
pip install -r requirements.txt



Project Structure
This project contains the following four directories, each implementing a different functionality:

1. Deal_Datas: Data Processing Module
This module contains scripts for data preprocessing and feature selection. The specific files are as follows:

feature_trait_analysis.py: Used to select SNP features related to a target trait.
feature_inter_analysis.py: Used to remove strongly correlated SNP features.
gene_snp.py: Extracts the correspondence between SNPs and genes.
gene_trans.py: Extracts selected gene expression features.
gene_relation.py: Extracts gene networks for selected features.
Once the paths are properly configured, you can run these scripts directly in the Python environment.

2. NetGP_Model: NetGP Model Module
This module contains the core source code for building and training the NetGP model. The files include:

NetGP_Datasets.py: Dataset reading and integration.
EarlyStopping.py: Implements early stopping logic.
SparseConnectedModule.py: Defines the GN layer logic.
GCN.py: Implements the extended GCN layer logic.
NetGP_Snp.py: Implements the NetGP(SNP) model.
NetGP_Trans.py: Implements the NetGP(Trans) model.
NetGP.py: Implements the NetGP(Trans+SNP) model.
train.py: Model training code.
predict.py: Model prediction code.
After configuring the paths, learning parameters, and other settings, you can run the main.py file to train and predict with the model.

3. Other_DL: Deep Learning Comparison Experiments Module
This module is used for deep learning model comparison experiments. The code is structured similarly to the NetGP_Model module. You can run deep learning experiments by configuring the necessary parameters and executing the main.py file.

4. Other_ML: Machine Learning Comparison Experiments Module
This module is used for machine learning model comparison experiments. The relevant code is executed in the ML.py file.

Usage
Configure paths and parameters: Before running the models, ensure that you have configured the paths and model parameters according to the project’s requirements.
Run the models: Run the main.py or ML.py files to start training, testing, and prediction processes for the respective models.

