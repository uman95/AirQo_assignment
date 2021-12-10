from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Timeseries Prediction')
parser.add_argument('--model', type=str, default='NN',
                    help='Baseline Model to train: any of the following {NN, LSTM, RNN}')
parser.add_argument('--data_Id', type=int, default=912223,
                    help='Dataset to be used: input is the channel ID of the data')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='training batch size (default: 1024)')
parser.add_argument('--test_batch_size', type=int, default=64,
                    help='testing batch size (default: 64)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train for (Default: 150)')
parser.add_argument('--target', type=list, default=['pm2_5'],
                    help='List of output target to predict at once')

cfg = parser.parse_args()
