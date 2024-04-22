import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# The evaluation metric used is Ranked Probability Score (RPS) (Constantinou & Fenton, 2013; Epstein, 1969), which is given by:

class RPS_loss(torch.nn.Module):
    def __init__(self):
        super(RPS_loss, self).__init__()

    def forward(self, x, t):
        diff1 = (x[:, 0] - t[:, 0]).unsqueeze(1)
        diff2 = ((x[:, 0] + x[:, 1]) - (t[:, 0] + t[:, 1])).unsqueeze(1)
        diff1_squared = torch.square(diff1)
        diff2_squared = torch.square(diff2)
        diff_sum = 0.5 * (diff1_squared + diff2_squared)
        # ranked probability score
        rps = torch.mean(diff_sum)
        return rps


def test_model(model, test_dataset, criterion, batch_size, device):
    print(f"Testing on: {device}")
    model.eval()
    model = model.to(device=device)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    with torch.no_grad():
        correct, total = 0, 0
        for i, (_h, _a, home_features, away_features, targets) in enumerate(dataloader):
            home_features = home_features.to(device)
            away_features = away_features.to(device)
            targets = targets.to(device)

            z = model(home_features, away_features)
            total_loss += criterion(z, targets)

            y = torch.argmax(z, axis=1)
            t = torch.argmax(targets, axis=1)

            correct += int(torch.sum(t == y))
            total += 1

        return correct / total, total_loss / len(dataloader)


# taken and adpated from lab7

def accuracy(model, dataset, batch_size, max=1000, device=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if device is None:
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        if torch.cuda.is_available():
            device = 'cuda'

    model.eval()
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
            total += 1
            if i >= max:
                break
        return correct / total


# modified code below is taken from csc413 lab 7
def train_model(
        model,
        criterion,
        train_dataset,  # training loader
        val_dataset,
        batch_size=100,
        learning_rate=0.001,
        num_epochs=20,
        plot_every=50,  # how often (in # iterations) to track metrics
        plot=True,
        accumulation_steps=10,
        device=None):  # whether to plot the training curve

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    if device is None:
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        if torch.cuda.is_available():
            device = 'cuda'

    model = model.to(device=device)
    criterion = criterion.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.1,patience=3, min_lr=1e-5)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0  # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (home_teams, away_teams, home_features, away_features, targets) in enumerate(train_loader):
                home_features = home_features.to(device)
                away_features = away_features.to(device)
                targets = targets.to(device)

                model.train()

                z = model(home_features, away_features)

                loss = criterion(z, targets) / accumulation_steps
                loss.backward()  # propagate the gradients

                if iter_count % accumulation_steps == 0:
                    optimizer.step()  # update the parameters
                    # print("lr, ", scheduler.get_last_lr())

                    optimizer.zero_grad()  # clean up accumualted gradients

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_dataset, batch_size, device=device)
                    va = accuracy(model, val_dataset, batch_size, device=device)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
            # va = accuracy(model, val_dataset, batch_size)
            # scheduler.step(va)
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

    # return iters, train_loss, train_acc, val_acc
