
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# The evaluation metric used is Ranked Probability Score (RPS) (Constantinou & Fenton, 2013; Epstein, 1969), which is given by:

class RPS_loss(torch.nn.Module):
    def __init__(self):
         super(RPS_loss, self).__init__()

    def forward(self, x, t):
        cum_output = torch.cumsum(x, dim=-1) # cumulative sums of predictions
        cum_target = torch.cumsum(t, dim=-1) # cumulative sums of targets
        rps = torch.mean(torch.sum(torch.square(cum_output - cum_target), dim= -1), dim= -1)
        return rps

# taken and adpated from lab7

def accuracy(model, dataloader, max=1000):

    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'


    with torch.no_grad():
        correct, total = 0, 0
        for i, (_h, _a, home_features, away_features, targets) in enumerate(dataloader):
            home_features = home_features.to(device)
            away_features = away_features.to(device)
            targets = targets.to(device)
            
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
                model,
                criterion,
                train_loader,           # training loader
                val_loader,
                learning_rate=0.001,
                num_epochs=20,
                plot_every=50,        # how often (in # iterations) to track metrics
                plot=True):           # whether to plot the training curve


    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'

    model = model.to(device=device)
    criterion = criterion.to(device=device)

    model.train() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (home_teams, away_teams, home_features, away_features, targets) in enumerate(train_loader):
                home_features = home_features.to(device)
                away_features = away_features.to(device)
                targets = targets.to(device)


                z = model(home_features,away_features)
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

