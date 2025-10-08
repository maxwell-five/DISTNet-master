# DISTGNN: Distinguishable Spatio-Temporal Graph Neural Network for Traffic Forecasting

This is a official code release of DISTGNN.

This code is mainly based on [BasicTS](https://github.com/zezhishao/BasicTS),

some codes are from [STD_MAE](https://github.com/Jimmy-7664/STD_MAE/blob/main/stdmae).



## Dependencies

### OS

Linux systems (*e.g.* Ubuntu and CentOS).

### Requirements

The code is built on Python 3.8, PyTorch 1.11, and [EasyTorch](https://github.com/cnstark/easytorch). Ensuring that PyTorch is installed correctly, you can install other dependencies via:

```pip
pip install -r requirements.txt
```



## Data Preparation

You can download data from [BasicTS](https://github.com/zezhishao/BasicTS/tree/master) and unzip it.

### Preparing Data

You can pre-process all datasets by

```
cd /path/to/your/project
bash scripts/data_preparation/all.sh
```

Then the `dataset` directory will look like this:

```
datasets
   ├─PEMS03
   ├─PEMS04
   ├─PEMS07
   ├─PEMS08
   ├─PEMS-BAY
   ├─METR-LA
   ├─raw_data
   |    ├─PEMS03
   |    ├─PEMS04
   |    ├─PEMS07
   |    ├─PEMS08
   |    ├─PEMS-BAY
   |    ├─METR-LA
   ├─README.md
```



## Training

### Pre-training on LTSRE

For example,run the following command to pre-training the dataset PEMS03.

```
python experiments/train.py --cfg='baselines/DISTGNN/LTSRE_PEMS03.py' --gpus='0'
```

After pre-training , copy your pre-trained best checkpoint to `mask_save/`.

### Training on DISTGNN

For example,run the following command to training the dataset PEMS03.

```
python experiments/train.py --cfg='baselines/DISTGNN/DISTGNN_PEMS03.py' --gpus='0'
```

And all the checkpoints and log will be saved in `checkpoints` directory after training.

### Evaluation

```
python experiments/inference.py --cfg='baselines/DISTGNN/DISTGNN_PEMS03.py' --gpus='0'
```




