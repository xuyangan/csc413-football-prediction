# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# %%
class LSTM(nn.Module):
    def __init__(self, batch_size, hidden_size, num_features, num_classes, num_heads=1):
        super(LSTM, self).__init__()
        self.batch_size     = batch_size
        self.hidden_size    = hidden_size
        self.num_features   = num_features
        self.num_classes    = num_classes
        self.num_heads      = num_heads


        self.lstm_A = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        self.lstm_B = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        self.attn_A = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.attn_B = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)


        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # input size is (2 * batch_size, seq_length, num_features)
        seq_A = x[:self.batch_size,:] # (batch_size, seq_length, num_features)
        seq_B = x[self.batch_size:,:] # same ^

        h_A, _ = self.lstm_A(seq_A) # (batch_size, seq_length, hidden_size)
        h_B, _ = self.lstm_B(seq_B) # same ^

        self_attn_A, _ = self.attn_A(h_A, h_A, h_A)
        self_attm_B, _ = self.attn_A(h_B, h_B, h_B)

        h_AB = torch.cat([self_attn_A[:,-1,:], self_attm_B[:,-1,:]], dim=1) # extract last hidden

        z = self.softmax(self.fc(h_AB))

        return z

# %%
# modified code below is taken from csc413 lab 7
def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.001,
                batch_size=100,
                num_epochs=10,
                plot_every=50,        # how often (in # iterations) to track metrics
                plot=True):           # whether to plot the training curve
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (texts, labels) in enumerate(train_loader):
                z = model(texts) # TODO

                loss = criterion(z, labels) # TODO

                loss.backward() # propagate the gradients
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data)
                    va = accuracy(model, val_data)
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
# %%


batch_size = 25
seq_length = 7
num_features = 15

model = LSTM(batch_size = batch_size, hidden_size=10, num_features=num_features, num_classes=3)

random_data = torch.rand((2 * batch_size, seq_length, num_features))

z = model.forward(random_data)
print(z)



# %%
