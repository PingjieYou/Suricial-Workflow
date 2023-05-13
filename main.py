import configs
import argparse
from train import train_temporal_memory_bank


def get_parase():
    parser = argparse.ArgumentParser(description="Train SVRCNet")
    parser.add_argument('--clip_size', default=configs.clip_size, type=int, help='sequence length, default 4')
    parser.add_argument('--batch_size', default=configs.batch_size, type=int, help='batch size, default 8')
    parser.add_argument('--epochs', default=configs.epochs, type=int, help='epochs to train and val, default 25')
    parser.add_argument('--lr', default=configs.learning_rate, type=float, help='learning rate for optimizer, default 1e-3')
    parser.add_argument('--optimizer', default="adam", help="optimizer for training, default adam")
    parser.add_argument('--momentum', default=configs.momentum, type=float, help='momentum for sgd, default 0.9')
    parser.add_argument('--weight_decay', default=configs.weigth_decay, type=float, help='weight decay for sgd, default 0')
    parser.add_argument('--dampening', default=configs.dampenning, type=float, help='dampening for sgd, default 0')
    parser.add_argument('--nesterov', default=configs.nesterov, type=bool, help='nesterov momentum, default False')
    parser.add_argument('--sgd_adjust', default=configs.sgd_adjust, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
    parser.add_argument('--sgd_step', default=configs.sgd_step, type=int, help='number of steps to adjust lr for sgd, default 5')
    parser.add_argument('--sgd_gamma', default=configs.sgd_gamma, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
    parser.add_argument('--alpha', default=configs.alpha, type=float, help='kl loss ratio, default 1.0')

    opts = parser.parse_args()
    return opts


def main():
    opts = get_parase()

    train_temporal_memory_bank.train(opts)


if __name__ == "__main__":
    main()
