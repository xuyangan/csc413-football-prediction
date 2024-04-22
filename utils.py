import torch

def get_last_inception_output_size(inception_out, inception_depth):
    return inception_out * 4 * (2 ** (inception_depth - 1))

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))