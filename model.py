from transformers import BertModel
import torch
import torch.nn as nn 


class SentimentAnalysis(nn.Module):
    
    def __init__(self,
                 in_features=768,
                 out_features=1,
                 pt_model_name="bert-base-multilingual-cased"):
        """Model for polar sentiment analysis on sequences.
        
        Uses a pre-trained BERT model and a single layer MLP.
        
        Activation function: sigmoid"""
        
        super(SentimentAnalysis, self).__init__()
        
        self.model = BertModel.from_pretrained(pt_model_name)
        
        self._fc = nn.Linear(in_features, out_features)
        
        self._activation = nn.Sigmoid()

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input):
        input_ids = torch.squeeze(input['input_ids'])
        
        output = self.model(input_ids=input_ids, 
                            attention_mask=input['attention_mask'],
                            output_hidden_states=True)
        
        X = torch.mean(output.last_hidden_state, dim=(1))
        X = self._fc(X)
        X = self._activation(X)
        return X