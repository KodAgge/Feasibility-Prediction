import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from plotnine import *

from lightgbm import LGBMClassifier

from skorch import NeuralNet
from skorch import NeuralNetRegressor


class Encoder(nn.Module):
    """Encoder component of autoencoder.

    Inspired by
        https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb#scrollTo=O9pIrJ674ki0
        
    Parameters
    ----------
    layers : list of integers
        Nodes of the hidden layers. No default.
    D : integer
        The dimensionality of the feature set, number of features. No default.
    act_fn : torch.nn callable
        Activation function from torch.nn package. No default.
    """
    def __init__(self, layers, D, act_fn):
        super().__init__()
        self.layers = layers
        self.D = D

        seqs = [nn.Sequential(
            nn.Linear(self.D, self.layers[0]),
            act_fn
        )]
        for i in range(len(self.layers)-1):
            nodes = self.layers[i]
            nnodes = self.layers[i + 1]
            seqs.append(nn.Sequential(
                nn.Linear(nodes, nnodes),
                act_fn
            ))
            

        self.encode = nn.Sequential(*seqs)
        
    def forward(self, X):
        """Forward pass in the network.
        
        Parameters
        ----------
        X : numpy.ndarray, float32
            Feature set. No default.
        """    
        encoded = self.encode(X)
        return encoded
    

class Decoder(nn.Module):
    """Decoder component of autoencoder.

    Inspired by:
        https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb#scrollTo=O9pIrJ674ki0
        
    Parameters
    ----------
    layers : list of integers
        Nodes of the hidden layers. No default.
    D : integer
        The dimensionality of the feature set. No default.
    act_fn : torch.nn callable
        Activation function from torch.nn package. No default.
    """
    def __init__(self, layers, D, act_fn):
        super().__init__()
        self.D = D
        self.layers = layers
        seqs = [nn.Sequential(
            nn.Linear(self.layers[0], self.D)
        )]
        for i in range(len(self.layers)-1):
            nodes = self.layers[i]
            nnodes = self.layers[i + 1]
            seqs.insert(0, nn.Sequential(
                nn.Linear(nnodes, nodes),
                act_fn
            ))
        self.decode = nn.Sequential(*seqs)
        
    def forward(self, X):
        """Forward pass in network.
        
        Parameters
        ----------
        X : numpy.ndarray, float32
            Feature set. No default.
        """
        decoded = self.decode(X)
        return decoded

    
class AutoEncoder(nn.Module):
    """Autoencoder, bundling the encoder and decoder.
    
    Parameters
    ----------
    layers : list of integers
        Nodes of the hidden layers.
    D : integer
        The dimensionality of the feature set. No default.
    act_fn : torch.nn callable
        Activation function from torch.nn package. No default.
    sparse : bool
        If True, generates a sparse autoencoder
    """
    def __init__(self, layers: list = [100, 40, 10], 
                 D = None, act_fn = nn.ReLU(), 
                 sparse: bool = False):
        super().__init__()
        self.D = D
        self.layers = layers

        self.encoder = Encoder(
            layers=self.layers, 
            D=self.D,
            act_fn=act_fn
        )
        self.decoder = Decoder(
            layers=self.layers,
            D=self.D,
            act_fn=act_fn 
        )
        
    def forward(self, X):
        """Forward pass in the network.
        
        Parameters
        ----------
        X : numpy.ndarray, float32
            Feature set. No default.
        """
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded
    

class AE(NeuralNetRegressor):
    """Wrapper to enable sparsity option for loss."""
    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        """Generate the loss of the autoencoder
        
        Parameters
        ----------
        y_pred : numpy.ndarray
            Decoded and encoded feature set. No default.
        y_true : numpy.ndarray
            Actual feature set. No default.
        X : numpy.ndarray
            Actual feature set. No default
        """
        decoded, encoded = y_pred
        sparse = None # np.unique(X['sparse'].numpy())[0]
        loss_reconstruction = super().get_loss(decoded, y_true, *args, **kwargs)
        l1 = 1e-3 * torch.abs(encoded).sum()
        return loss_reconstruction + l1 if sparse else loss_reconstruction

    
class AutoencoderLGBMClassifier():
    """ Autoencoder + LGBMClassifier
    
    Class for combining dimensionlity reduction and classification.
    Follows the `sklearn` API.
    
    Paremeters
    ----------
    D : int
        Number of features (dimensions). The default is 3851, the number of dimensions in the dataset used.
    max_depth : int
        Maximum depth allowed for each tree. The default is 100.
    max_iter : int
        Number of iterations allowed for classifier. The defualt is 500
    layers : list of ints
        Number of nodes in each layer. List length = number of layers. The default is [2056, 1024, 256] 
    lr : float
        Autoencoder learning rate. The default is 0.1.
    act_fn : torch.nn layer
        Activation function for all layers. The same function is used for all layers. The default is nn.LeakyReLU()
    batch_size : int
        Batch size for epochs. The default is 5000.
    max_epochs : int
        Number of epochs. The default is 100.
        
    """

    def __init__(self, D=3851, max_depth = 100, max_iter = 500,
                 layers = [2056, 1024], num_leaves = 31,
                 lr = 1.4, act_fn = nn.LeakyReLU(),
                 batch_size = 2000, max_epochs = 50,
                 verbosity=-1, plot_loss = False):
        self.lgbmclassifier = LGBMClassifier
        self.clf_parameters = {
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'max_iter': max_iter,
            'verbosity': verbosity
        }

        self.autoencoder = AE
        self.reducer = None

        self.autoencoder_parameters = {
            'module__layers': layers,
            'lr': lr,
            'module__D': D,
            'module__act_fn': act_fn,
            'batch_size': batch_size,
            'max_epochs': max_epochs
        }
        # To save the train val plots
        self.plot_loss = plot_loss

    def fit(self, x, y):
        """Custom fit method.
        
        Parameters
        ----------
        x : numpy.ndarray
            Feature set. No default
        y : numpy.ndarray
            Target labels. No default.
        """
        self.reducer = self.autoencoder(
            AutoEncoder,
            verbose = 0,
            **self.autoencoder_parameters
        )
        # Dimensionality reduction
        x = x.astype(np.float32)
        self.reducer.fit(x, x)
        if self.plot_loss: print(self.plot_train_val_loss())
        _, x_reduced = self.reducer.forward(x)
        
        # Classification
        self.clf = self.lgbmclassifier(**self.clf_parameters)
        self.clf.fit(x_reduced, y)

        
    def predict(self, x):
        """Custom predict method.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature set. No default
        """
        _, x_reduced = self.reducer.forward(x.astype(np.float32))
        y_pred = self.clf.predict(x_reduced)

        return y_pred
    

    def predict_proba(self, x):
        """"Custom predict probability.
        
        Parameters
        ----------
        x : numpy.ndarray
            Feature set. No default.
        """
        _, x_reduced = self.reducer.forward(x.astype(np.float32))
        y_scores = self.clf.predict_proba(x_reduced)
        
        return y_scores

    
    def get_autoencoder(self):
        """Return the fitted autoencoder."""
        return self.reducer
    
    
    def plot_train_val_loss(self, file_name = None):
        """Plotting reconstruction loss in train and validation.
        
        Parameters
        ----------
        file_name : str
            If not None, then plot is saved to that specified by file_name. The default is None
        """
        train_val_loss = self.reducer.history[:, ('train_loss', 'valid_loss')]
        loss_df = (pd.DataFrame(train_val_loss)
                 .reset_index()
                 .rename(columns = {
                     0: 'Train', 
                     1: 'Validation',
                     'index': 'Epoch'
                 })
              )
        loss_df_melted = pd.melt(loss_df, value_vars = ["Train", "Validation"], id_vars="Epoch")
        ## TODO: Change to plotnine
        loss_plot = (
            ggplot(loss_df_melted, aes(x = 'Epoch', y = 'value', color = 'variable')) + 
            geom_line() + 
            theme(legend_title=element_blank()) +
            ylab("Reconstruction loss")
        )
        if file_name: ggsave(loss_plot, filename = './misc/' + file_name)
        return loss_plot
