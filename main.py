import argparse
import sys
from pathlib import Path
import torch
from typing import Optional
from utils import save_model, load_model
from models.train_model import RPS_loss, train_model, test_model
from models.lstm import LSTM
from models.transformer import Transformer
from train import get_test_data, get_train_val_data


# Namespace(command='train', 
#           lr=1e-05, 
#           split=0.8, 
#           epochs=5, 
#           gradient_accumulation=5, 
#           device=device(type='cpu'), 
#           inception_depth=2, 
#           inception_out_dim=32, 
#           inception_residual=True, 
#           inception_bottleneck_dim=32, 
#           lstm_layers=2, 
#           transformer_layers=6, 
#           lstm_heads=1, 
#           transformer_heads=1, 
#           hidden_size=512, 
#           transformer_ff_size=256, 
#           dropout=0.1, 
#           batch_size=128, 
#           model_type='transformer', 
#           loss='rps')

def train(opts):
    train_data, val_data = get_train_val_data(opts.split)
    time_dim = train_data[0][2].size(0)
    num_features = train_data[0][2].size(1)
    output_classes = train_data[0][4].size(0)

    if opts.loss == 'rps':
        criterion = RPS_loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opts.model_type == 'lstm':
        model = LSTM(
            num_features, 
            inception_depth=opts.inception_depth, 
            inception_out=opts.inception_out_dim, 
            lstm_num_layers=opts.lstm_layers, 
            hidden_size=opts.hidden_size, 
            num_heads=opts.lstm_heads, 
            num_classes=output_classes, 
            bottleneck_dim=opts.inception_bottleneck_dim
        )
        print(f'========= Parameters - {opts.model_type} =========')
        print(f'Inception-> ')
        print(f'\t Depth: {opts.inception_depth}')
        print(f'\t Out Dim: {opts.inception_out_dim}')
        print(f'\t Bottleneck: {opts.inception_bottleneck_dim}')
        print(f'\t Residual: {opts.inception_residual}')
        print(f'{opts.model_type}->')
        print(f"\t Num layers: {opts.lstm_layers}")
        print(f"\t Hidden layer size: {opts.hidden_size}")
        print(f"\t Attention heads: {opts.lstm_heads}")
        print(f'Training parameters->')
        print(f"\t loss: {opts.loss}")
        print(f'\t lr: {opts.lr}')
        print(f'\t Accumulate grads: {opts.gradient_accumulation}')
        print(f'\t Batch size: {opts.batch_size}')
        print(f'\t Epochs: {opts.epochs}')
    else:
        model = Transformer(
            num_features=num_features, 
            inception_depth=opts.inception_depth, 
            inception_out=opts.inception_out_dim, 
            num_heads=opts.transformer_heads, 
            num_layers=opts.transformer_layers, 
            d_ff=opts.transformer_ff_size, 
            d_h=opts.hidden_size, 
            max_timespan=time_dim,
            dropout=opts.dropout, 
            inception_use_residual=opts.inception_residual, 
            inception_bottleneck=opts.inception_bottleneck_dim)
        
        print(f'========= Parameters - {opts.model_type} =========')
        print(f'Inception-> ')
        print(f'\t Depth: {opts.inception_depth}')
        print(f'\t Out Dim: {opts.inception_out_dim}')
        print(f'\t Bottleneck: {opts.inception_bottleneck_dim}')
        print(f'\t Residual: {opts.inception_residual}')
        print(f'{opts.model_type}->')
        print(f"\t Num layers: {opts.transformer_layers}")
        print(f"\t Hidden layer size: {opts.hidden_size}")
        print(f"\t FF Hidden layer size: {opts.transformer_ff_size}")
        print(f"\t Attention heads: {opts.transformer_heads}")
        print(f"\t Dropout: {opts.dropout}")
        print(f'Training parameters->')
        print(f"\t loss: {opts.loss}")
        print(f'\t lr: {opts.lr}')
        print(f'\t Accumulate grads: {opts.gradient_accumulation}')
        print(f'\t Batch size: {opts.batch_size}')
        print(f'\t Epochs: {opts.epochs}')
        
    if opts.device == 'gpu':
        if torch.backends.mps.is_available():
            opts.device = 'mps'
        if torch.cuda.is_available():
            opts.device = 'cuda'
    
    print(f"Training on: {opts.device}")

    train_model(model=model,
                criterion=criterion,
                train_dataset=train_data,           # training loader
                val_dataset=val_data,
                batch_size = opts.batch_size,
                learning_rate=opts.lr,
                num_epochs=opts.epochs,
                plot_every=50,        # how often (in # iterations) to track metrics
                plot=False,
                accumulation_steps=opts.gradient_accumulation,
                device=opts.device)
    
    save_model(model, f"trained_models/{opts.model_type}_{opts.loss}_{opts.lr}.pth")

def test(opts):
    if opts.device == 'gpu':
        if torch.backends.mps.is_available():
            opts.device = 'mps'
        if torch.cuda.is_available():
            opts.device = 'cuda'

    if opts.loss == 'rps':
        criterion = RPS_loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    test_data = get_test_data()
    model = load_model(opts.model_path)
    
    accuracy, loss = test_model(model, 
                                test_dataset=test_data, 
                                criterion=criterion, 
                                batch_size=opts.batch_size, 
                                device=opts.device)
    print(f"Test accuracy: {accuracy}")
    print(f"Test loss: {loss}")

    
def main(args: Optional[list[str]] = None) -> int:
    parser = build_parser()
    opts = parser.parse_args(args)

    torch.manual_seed(42)

    if opts.command == "train":
        train(opts)
    elif opts.command == "test":
        test(opts)
        
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(help="Specific commands", dest="command")
    build_training_parser(subparsers)
    build_testing_parser(subparsers)
    return parser


def build_training_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    
    parser = subparsers.add_parser("train", help="Train a model")
    
    parser.add_argument(
        '--save-model',
        type=bool,
        default=False,
        help='Whether to save model to disk after training'
    )

    parser.add_argument(
        "--lr",
        type=proportion,
        default=1e-5,
        help="Training learning rate"
    )

    parser.add_argument(
        "--split",
        type=proportion,
        default=0.8,
        help="Training dataset train/val split percentage"
    )

    parser.add_argument(
        "--epochs",
        type=lower_bound,
        metavar="E",
        default=5,
        help="The number of epochs to run in total.",
    )

    parser.add_argument(
        "--gradient-accumulation",
        metavar="G",
        type=lower_bound,
        default=5,
        help="The number of mini-batches to process at once",
    )

    parser.add_argument(
        "--device",
        metavar="DEV",
        type=str,
        default="cpu",
        choices=['cpu', 'gpu'],
        help='Where to do training (e.g. "cpu", "gpu")',
    )

    # Inception hyperparameters
    parser.add_argument(
        "--inception-depth",
        type=lower_bound,
        default=2,
        help='Depth of inception network'
    )

    parser.add_argument(
        '--inception-out-dim',
        type=lower_bound,
        default=32,
        help='Number of features for each inception filter'
    )

    parser.add_argument(
        '--inception-residual',
        type=bool,
        default=True,
        help='Whether to use residual connections in inception network'
    )

    parser.add_argument(
        '--inception-bottleneck-dim',
        type=none_or_int,
        default=32,
        help='Dimension of bottleneck layer in inception'
    )

    num_layers = parser.add_mutually_exclusive_group()
    num_heads = parser.add_mutually_exclusive_group()

    # LSTM and Transformer Hyperparameters
    num_layers.add_argument(
        '--lstm-layers', 
        type=lower_bound,
        default=2,
        help="Number of hidden layers in LSTM"
    )

    num_layers.add_argument(
        '--transformer-layers',
        type=lower_bound,
        default=6,
        help="Number of transformer blocks"
    )

    num_heads.add_argument(
        "--lstm-heads",
        metavar="N",
        default=1,
        type=lower_bound,
        help="The number of heads to use for the multi-head attention mechanism in LSTM",
    )

    num_heads.add_argument(
        "--transformer-heads",
        metavar="N",
        default=8,
        type=lower_bound,
        help="The number of heads to use for the multi-head attention mechanism in Transformer",
    )

    parser.add_argument(
        '--hidden-size',
        default=512,
        type=lower_bound,
        help="The hidden state size in LSTM cell or Transformer prediction network"
    )

    parser.add_argument(
        "--transformer-ff-size",
        metavar="H",
        type=lower_bound,
        default=256,
        help="The hidden state size in one direction inside the Transformer",
    )

    parser.add_argument(
        "--dropout",
        metavar="p",
        type=proportion,
        default=0.1,
        help="The probability of dropping an encoder hidden state during training",
    )

    add_common_model_options(parser)
    return parser


def build_testing_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("test", help="Evaluate a model")

    parser.add_argument(
        "--device",
        metavar="DEV",
        type=str,
        default="cpu",
        choices=['cpu', 'gpu'],
        help='Where to do testing (e.g. "cpu", "gpu")',
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path of the saved model",
    )

    add_common_model_options(parser)
    return parser


def add_common_model_options(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--batch-size",
        metavar="B",
        type=lower_bound,
        default=128,
        help="The number of sequences to process at once",
    )

    parser.add_argument(
        "--model-type", 
        type=str,
        choices=['lstm', 'transformer'],
        default='lstm',
        help="Type of the model")
    
    parser.add_argument(
        "--loss",
        type=str,
        choices=['rps', 'ce'],
        default='rps',
        help='Loss function to use for training or evaluation'
    )

def lower_bound(v: str, low: int = 1) -> int:
    v = int(v)
    if v < low:
        raise argparse.ArgumentTypeError(f"{v} must be at least {low}")
    return v

def proportion(v: str, inclusive: bool = True) -> float:
    v = float(v)
    if inclusive:
        if v < 0.0 or v > 1.0:
            raise argparse.ArgumentTypeError(f"{v} must be between [0, 1]")
    else:
        if v <= 0 or v >= 1:
            raise argparse.ArgumentTypeError(f"{v} must be between (0, 1)")
    return v

def none_or_int(v: str) -> int|None:
    if v == 'None':
        return None
    return int(v)

if __name__ == "__main__":
    sys.exit(main())
