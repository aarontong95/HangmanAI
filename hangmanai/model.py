import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Fun

from hangmanai.config import N_TOKENS, HIDDEN_DIM, LAYER_DIM, N_LABELS, LEARNING_RATE, NUMBER_OF_THREADS, IGNORE_INDEX

torch.set_num_threads(NUMBER_OF_THREADS)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x = Fun.one_hot(x, num_classes=self.input_dim).type(torch.FloatTensor)
        h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)

        return out

def load_model(path):
    model = LSTMModel(N_TOKENS, HIDDEN_DIM, LAYER_DIM, N_LABELS)
    model.load_state_dict(torch.load(path))

    return model

def train(train_loader, model=None, save_path=None):
    if not model:
        model = LSTMModel(N_TOKENS, HIDDEN_DIM, LAYER_DIM, N_LABELS)
    error = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    count = 0

    for images, labels in train_loader:
        train  = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(train)
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1)

        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        count += 1

    if save_path:
        torch.save(model.state_dict(), save_path)

    return model

