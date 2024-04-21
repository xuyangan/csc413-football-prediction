import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.layers.InceptionTimeBlock import InceptionTimeBlock as Inception

class LSTM(nn.Module):
    def __init__(self, num_features, out_channels, hidden_size, num_heads, num_classes, bottleneck_dim = None):
        super(LSTM, self).__init__()

        self.num_features   = num_features

        self.out_channels   = out_channels
        self.hidden_size    = hidden_size

        self.num_heads      = num_heads
        self.num_classes    = num_classes

        self.inception = Inception(num_features, out_channels, bottleneck_dim=bottleneck_dim)

        self.lstm = nn.LSTM(input_size= 4 * out_channels, hidden_size=hidden_size, batch_first=True)

        self.attn_H = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.attn_A = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.Softmax = nn.Softmax(dim=1)


    def forward(self, home, away):

        batch_size = home.size(0)
        seq_length = home.size(1)

        stacked_x = torch.stack((home, away)).reshape(2 * batch_size, seq_length, self.num_features)

        inception_features = self.inception(stacked_x)

        indices_H = torch.arange(0, batch_size)
        indices_A = torch.arange(batch_size, 2 * batch_size)

        seq_H = torch.index_select(inception_features, 0, indices_H) # (batch_size, seq_length, num_features)
        seq_A = torch.index_select(inception_features, 0, indices_A)

        h_H, _ = self.lstm(seq_H) # (batch_size, seq_length, hidden_size)
        h_A, _ = self.lstm(seq_A) # same ^

        self_attn_H, _ = self.attn_H(h_H, h_H, h_H)
        self_attn_A, _ = self.attn_A(h_A, h_A, h_A)

        code = torch.cat([self_attn_H[:,-1,:], self_attn_A[:,-1,:]], dim=-1) #  (batch_size, 2 * (hidden_size + 1))

        z = self.Softmax(self.fc(code))

        return z

# The evaluation metric used is Ranked Probability Score (RPS) (Constantinou & Fenton, 2013; Epstein, 1969), which is given by:

def RPS_loss(output, target):

    # output/target shape: (batch, r)

    cum_output = torch.cumsum(output, dim=-1) # cumulative sums of predictions
    cum_target = torch.cumsum(target, dim=-1) # cumulative sums of targets
    rps = torch.mean(torch.sum(torch.square(cum_output - cum_target), dim= -1), dim= -1)
    return rps


# taken and adpated from lab7

def accuracy(model, dataloader, max=1000):
    """
    Estimate the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model`   - An object of class nn.Module
        `dataset` - A dataset of the same type as `train_data`.
        `max`     - The max number of samples to use to estimate
                    model accuracy

    Returns: a floating-point value between 0 and 1.
    """

    correct, total = 0, 0
    for i, (_h, _a, home_features, away_features, targets) in enumerate(dataloader):
        z = model(home_features, away_features)
        y = torch.argmax(z, axis=1)
        t = torch.argmax(targets, axis=1)

        correct += int(torch.sum(t == y))
        total   += 1
        if i >= max:
            break
    return correct / total

# modified code below is taken from csc413 lab 7
def train_model(
                model,                # an instance of MLPModel
                criterion,
                train_loader,           # training loader
                val_loader,
                learning_rate=0.001,
                num_epochs=10,
                plot_every=50,        # how often (in # iterations) to track metrics
                plot=True):           # whether to plot the training curve


    criterion = RPS_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (home_teams, away_teams, home_features, away_features, targets) in enumerate(train_loader):

                z = model(home_features, away_features)
                loss = criterion(z, targets)

                loss.backward() # propagate the gradients
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_loader)
                    va = accuracy(model, val_loader)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
    finally:
        # This try/finally block is to display the training curve
        # even if training is interrupted
        if plot:
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend(["Train", "Validation"])

