# Implementation details of the evaluation of NormWear

## fullshot on CogLoad1
The task is considered a multi output regression one where signals are processed using [process_cogload.py](../utils/process_cogload.py). Then, they are fed to NormWear to generate the embedings. Then, to a shallow neural network whose architecture is presented in the table below
| Layer type | output shape |
| -------- | ------- |
| Dropout | (None,5376) |
| Batch Normalization | (None,5376) |
| Dense | (None,6) |

The backbone model (NormWear) was freezed only the small neural network was trained with this set of hyperparameters:
| Hyperparameter | value |
|--|--|
| batch size |64|
| decay rate|0.99|
|decay steps|49|
|dropout_rate|0.3|
| number of epochs|1000|
|initial learning rate|1e-3|
|learning rate scheduler| exponential|
| L2 regularization parameter| 1e-3|
| staircase learning rate decay | false|
| validation split|0.2|
the hyperparameter tuning was done using claude code by iteratively trying a set a parameters. The agent was prompted 
``` train the model on Cogload by optimizing the hyperparamers (including the ability to change the learning rate schedualer) to minize the mae while not overfiting. Here is the command to train : CUDA_VISIBLE_DEVICES=0 python3 -m NormWear.downstream_main --model_name normwear --model_weight_dir NormWear/weights/normwear_pretrain_ckpt.pth --group 1 --data_path NormWear/data --num_runs 1 --prepare_embed 0 --remark test_run ```
The result is **77.92%** in the mean relative error.