from DeBERTa import deberta
import torch.nn as nn
from typing import Dict, Any
import os

_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), ".." + os.sep + ".."))
_MODEL_PATH = os.path.join(_ROOT_PATH, 'models', 'deberta-base', 'pytorch_model.bin')


class Config:
    def __init__(self, config: Dict[str, Any]):
        self.embedding_dim = config.get('embedding_dim')
        self.lstm_hidden_dim = config.get('lstm_hidden_dim')
        self.lstm_num_layers = config.get('lstm_num_layers')
        self.num_tags = config.get('num_tags')


class DebertaNER(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # load pretrained Deberta model
        self.deberta = deberta.DeBERTa(pre_trained=_MODEL_PATH)
        self.deberta.apply_state()

        # the LSTM tokens embedded sentence
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.lstm_hidden_dim // 2,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(config.lstm_hidden_dim, config.num_tags)

        # softmax layer normalize values of equal to 1
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids):
        # apply the Deberta model as pretrained model. using the last layer of this model.
        encodings = self.deberta(input_ids)['hidden_states'][-1]

        # run the LSTM along the encoding with length batch_max_len
        # output dim: batch_size * batch_max_len * lstm_hidden_dim
        lstm_outputs, _ = self.lstm(encodings)

        # reshape the Variable so that each row contains one token
        # output dim: (batch_size*batch_max_len) * lstm_hidden_dim
        lstm_outputs = lstm_outputs.reshape(lstm_outputs.shape[0]*lstm_outputs.shape[1], lstm_outputs.shape[2])

        # apply the fully connected layer and obtain the output for each token
        # output dim: (batch_size*batch_max_len) * num_tags
        fc_outputs = self.fc(lstm_outputs)

        # normalize the hidden layer and the value of the labels equal to 1
        # output dim: (batch_size*batch_max_len) * num_tags
        outputs = self.softmax(fc_outputs)

        return outputs
