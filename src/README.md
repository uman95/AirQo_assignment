# Models

Here we have considered 3 machine learning models {Neural Network, Recurrent Neural Network and Long Short Term Memory} implementation with Tensorflow framework and train on each 42 different AirQo monitoring sites data.
To use this code, you need to be in the src directory and run
`python train.py` with the following options

```
usage: train.py --data_Id [data_channel_ID]
                --model [model to train, options are {'NN', 'LSTM', 'RNN'}]
                --epochs [number of training epoch]
                --batch_size [training batch size]
                --test_batch_size [test batch size]
                --target [target label to predict]

```

### Training

```
> python3 train.py --data_Id 912223 --model 'NN' --epoch 20
```
