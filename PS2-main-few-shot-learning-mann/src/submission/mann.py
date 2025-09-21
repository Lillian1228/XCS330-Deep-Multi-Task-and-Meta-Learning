import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###

        # split support and query inputs and labels
        support_input_images = input_images[:, :-1, :, :] # [B, K, N, 784]
        query_input_images = input_images[:, -1:, :, :] # [B, 1, N, 784]
        support_labels = input_labels[:, :-1, :, :] # [B, K, N, N]
        # pass 0 for the query set
        query_labels = torch.zeros_like(input_labels[:, -1:, :, :]) # [B, 1, N, N]
        
        # concatenate labels and images
        support_input = torch.cat((support_input_images, support_labels), dim=-1) # [B, K, N, 784+N]
        query_input = torch.cat((query_input_images, query_labels), dim=-1) # [B, 1, N, 784+N]
        inputs = torch.cat((support_input, query_input), dim=1) # [B, K+1, N, 784+N]

        B = input_images.shape[0]
        N = self.num_classes
        inputs = torch.reshape(inputs, (B, -1, input_images.shape[-1]+N))
        # sequantially pass inputs through network
        output_1, hc = self.layer1(inputs.float())
        output_2, hc = self.layer2(output_1)

        #input = torch.reshape(output_2, shape = (B, self.samples_per_class, N, N)) # not sure 
        input = output_2.view(B, self.samples_per_class, N, N)
        return input
        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        loss = None

        ### START CODE HERE ###
        preds_target = preds[:, -1, :, :].reshape(-1, preds.shape[-1])
        labels_target = labels[:, -1, :, :].reshape(-1, labels.shape[-1])      
        loss = F.cross_entropy(preds_target, labels_target)
 
        ### END CODE HERE ###

        return loss
