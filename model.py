import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from crf import CRF
from typing import Tuple

from train import *

START_TAG = "<START>"
STOP_TAG = "<STOP>"

device = torch.device("cpu")

class NERModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        tag_to_ix: dict,
        embedding_dim,
        hidden_dim,
        num_laters,
        dropout, 
        pre_word_embeds=None,
        use_gpu=False,
        use_crf=False,
    ):
        super(NERModel, self).__init__()
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_laters
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.start_tag_id = tag_to_ix[START_TAG]
        self.end_tag_id = tag_to_ix[STOP_TAG]
        self.dropout=nn.Dropout(p=0.1)
        self.word_embeds=nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers, bidirectional=True, batch_first = True)
        self.dense= nn.Linear(self.hidden_dim*2, self.tagset_size)      


    def _get_features(self, sentence: torch.Tensor):
        """

        This is the function to get the features of the sentences from the BiLSTM model.
        
        Args:
            sentence (torch.Tensor): The input sentence to be processed. The shape of the tensor is (batch_size, seq_len, embedding_size).

        Returns:
            torch.Tensor: The output of the BiLSTM model.
        """

        embedding= self.dropout(self.word_embeds(sentence))

        lstm_out, _ = self.lstm(embedding)
        lstm_out= self.dropout(lstm_out)
        #feeding it to the linear layer
        pred_tag_lstm = self.dense(lstm_out)
        return pred_tag_lstm
        

    def forward(self, sentence: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """
        The loss for BiLSTM-CRF model is the negative log likelihood of the model.
        The loss for BiLSTM model is the cross entropy loss.
        Args:
            sentence (torch.Tensor): The input sentence to be processed.
            tags (torch.Tensor): The ground truth tags of the input sentence.

        Returns:
            scores (torch.Tensor): The output of the model. It is the loss of the model.
        
           """
        features=self._get_features(sentence)
        features_final = features.view(features.shape[0]*features.shape[1], -1)
        tags_final = tags.view(features.shape[0]*features.shape[1])
        
        if self.use_crf:      
             # for bilstm crf model  
            score= CRF.forward(features_final, tags_final)
        
        else: 
            #for bilstm model

            criterion=nn.CrossEntropyLoss()

            score= criterion(features_final, tags_final)

        return score
        
        
        #raise NotImplementedError

    def inference(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        This is the function that will be called when the model is being tested.
        
        Args:
            sentence (torch.Tensor): The input sentence to be processed.

        Returns:
            The score and the predicted tags of the input sentence.
            score (torch.Tensor): The score of the predicted tags.
            tag_seq (torch.Tensor): The predicted tags of the input sentence.
            
        """
        features=self._get_features(sentence)     
        #bilstm crf pred
        if self.use_crf:
            score, tag_seq= CRF.inference(features)
            return score, tag_seq
        
        else:
            score, tag_seq = torch.max(features, 2)

        return score, tag_seq
            

        
        
        
        #raise NotImplementedError
