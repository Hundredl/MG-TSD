
import warnings
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import os
import copy
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from multi_gran_generator import creat_coarse_data, creat_coarse_data_elec
from mgtsd_estimator import mgtsdEstimator
from trainer import Trainer
from pathlib import Path
import wandb
import ast
from utils import plot
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='mgtsd', help='model name')
    parser.add_argument('--dataset', type=str,
                        default="solar", help='dataset name')
    parser.add_argument('--cuda_num', type=str,
                        default='0', help='cuda number')

    parser.add_argument('--result_path', type=str,
                        default='./results/', help='result path')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-05)
    parser.add_argument('--num_cells', type=int, default=128,
                        help='number of cells in the rnn')
    parser.add_argument('--diff_steps', type=int,
                        default=100, help='diff steps')

    parser.add_argument('--input_size', type=int, default=552,
                        help='the input size of the current dataset, which is different from the original feature size but can be calculated from the original feature size.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mg_dict', type=str, default='1_4',
                        help='the multi-granularity list, 1_4 means 1h and 4h, 1_4_8 means 1h, 4h and 8h')
    parser.add_argument('--num_gran', type=int, default=2,
                        help='the number of granularities, must be equal to the length of mg_dict')
    parser.add_argument('--share_ratio_list', type=str, default="1_0.9",
                        help='the share ratio list, 1_0.9, means that for the second granularity, 90% of the diffusion steps are shared with the finest granularity.')
    parser.add_argument('--weight_list', type=str, default="0.9_0.1",
                        help='the weight list, 0.9_0.1 means that the weight for the first granularity is 0.9 and the weight for the second granularity is 0.1.')
    parser.add_argument('--run_num', type=str, default="1",
                        help='the index of the run, used for the result file name')
    parser.add_argument('--wandb_space', type=str,
                        default="test", help='the space name of the wandb')
    parser.add_argument('--wandb_key', type=str, default="your wandb key",
                        help='the key of the wandb, please replace it with your own key')
    parser.add_argument('--log_metrics', type=str2bool, default="False",
                        help='whether to log the metrics to the wandb when training. it will slow down the training process')

    # 返回一个命名空间，包含传递给命令的参数
    return parser.parse_args()


alias = {
    'elec': 'electricity_nips',
    'wiki': 'wiki-rolling_nips',
    'cup': 'kdd_cup_2018_without_missing',
    'solar': 'solar_nips',
    'traf': 'traffic_nips',
    'taxi': 'taxi_30min'
}
input_size_all = {
    'solar': 552,
    'cup': 1084,
    'traf': 3856,
    'taxi': 7290,
    'elec': 1484,
    'wiki': 8002,
}
feature_size_all = {
    'fred': 107,
    'solar': 137,
    'cup': 270,
    'traf': 963,
    'taxi': 1214,
    'elec': 370,
    'wiki': 2000,
}

args = parse_args()
model_name = args.model_name
cuda_num = args.cuda_num
result_path = args.result_path
Path(result_path).mkdir(parents=True, exist_ok=True)
epoch = args.epoch
diff_steps = args.diff_steps
num_gran = args.num_gran

dataset_name = args.dataset
input_size = input_size_all[dataset_name]
args.input_size = input_size
batch_size = args.batch_size
mg_dict = [float(i) for i in str(args.mg_dict).split('_')]
print(f"mg_dict:{mg_dict}")
share_ratio_list = [float(i) for i in str(args.share_ratio_list).split('_')]
weight_list = [float(i) for i in str(args.weight_list).split('_')]
weights = weight_list
print(f"share_ratio_list:{share_ratio_list}")
learning_rate = args.learning_rate
num_cells = args.num_cells
if args.log_metrics:
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_space, save_code=True, config=args)
print(args)

device = torch.device(
    f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
dataset = get_dataset(alias[dataset_name], regenerate=False)
print(dataset.metadata.feat_static_cat[0].cardinality)


train_grouper = MultivariateGrouper(max_target_dim=min(
    2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)),
                                   max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
print("================================================")
print("prepare the dataset")


test_data = dataset.test
train_data = dataset.train
if dataset_name == 'cup':  # cup dataset has different length of the target, need to pad the target to the same length
    test_data = list(test_data)
    for i in range(len(test_data)):
        if len(test_data[i]['target']) == 10898:
            test_data[i]['target'] = np.concatenate(
                (test_data[i]['target'], np.zeros(8)), axis=0)
    dataset_test = test_grouper(test_data)
else:
    dataset_test = test_grouper(test_data)

dataset_train = train_grouper(train_data)

if dataset_name == 'elec':
    data_train, data_test = creat_coarse_data_elec(dataset_train=dataset_train,
                                                   dataset_test=dataset_test,
                                                   mg_dict=mg_dict)
else:
    data_train, data_test = creat_coarse_data(dataset_train=dataset_train,
                                              dataset_test=dataset_test,
                                              mg_dict=mg_dict)
print("================================================")
print("initlize the estimator")


estimator = mgtsdEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    context_length=dataset.metadata.prediction_length,
    cell_type='GRU',
    input_size=input_size,
    freq=dataset.metadata.freq,
    loss_type='l2',
    scaling=True,
    diff_steps=diff_steps,
    share_ratio_list=share_ratio_list,
    beta_end=0.1,
    beta_schedule="linear",
    weights=weights,
    num_cells=num_cells,
    num_gran=num_gran,
    trainer=Trainer(device=device,
                    epochs=epoch,
                    learning_rate=learning_rate,
                    num_batches_per_epoch=100,
                    batch_size=batch_size,
                    log_metrics=args.log_metrics,)
)
print("================================================")
print("start training the network")
predictor = estimator.train(
    data_train, num_workers=8, validation_data=data_test)

print("===============================================")
print("make predictions")
forecast_it, ts_it = make_evaluation_predictions(dataset=data_test,
                                                 predictor=predictor,
                                                 num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

print("the number of days for targets")
print(len(targets))
print(forecasts[0].samples.shape)  # (100, 24, 137)
print(targets[0].shape)  # (7177, 274)

targets_list = []
forecasts_list = []
target_dim = estimator.target_dim
target_columns = targets[0].iloc[:, :target_dim].columns
for cur_gran_index, cur_gran in enumerate(mg_dict):
    targets_cur = []
    predict_cur = []
    predict_cur = copy.deepcopy(forecasts)

    for i in range(len(targets)):
        targets_cur.append(
            targets[i].iloc[:, (cur_gran_index * target_dim):((cur_gran_index + 1) * target_dim)])
        targets_cur[-1].columns = target_columns
    for day in range(len(forecasts)):
        predict_cur[day].samples = forecasts[day].samples[:, :,
                                                          (cur_gran_index * target_dim):((cur_gran_index + 1) * target_dim)]
        print(f'predict_cur:{predict_cur[day].samples.shape}')
    targets_list.append(targets_cur)
    forecasts_list.append(predict_cur)


# Ignore all warnings
warnings.filterwarnings("ignore")


agg_metric_list = []
for cur_gran_index, cur_gran in enumerate(mg_dict):
    evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:],
                                      target_agg_funcs={'sum': np.sum})
    agg_metric, item_metrics = evaluator(targets_list[cur_gran_index], forecasts_list[cur_gran_index],
                                         num_series=len(data_test)/2)
    agg_metric_list.append(agg_metric)


for cur_gran_index, cur_gran in enumerate(mg_dict):
    agg_metric = agg_metric_list[cur_gran_index]
    print(f"=======evaluation metrics for {cur_gran} h samples")
    print("CRPS:", agg_metric["mean_wQuantileLoss"])
    print("ND:", agg_metric["ND"])
    print("NRMSE:", agg_metric["NRMSE"])
    print("")
    print("CRPS-Sum:", agg_metric["m_sum_mean_wQuantileLoss"])
    print("ND-Sum:", agg_metric["m_sum_ND"])
    print("NRMSE-Sum:", agg_metric["m_sum_NRMSE"])

    CRPS = agg_metric["mean_wQuantileLoss"]
    ND = agg_metric["ND"]
    NRMSE = agg_metric["NRMSE"]
    CRPS_Sum = agg_metric["m_sum_mean_wQuantileLoss"]
    ND_Sum = agg_metric["m_sum_ND"]
    NRMSE_Sum = agg_metric["m_sum_NRMSE"]
    if args.log_metrics:
        wandb.log({f'CRPS_Sum_{cur_gran}': CRPS_Sum,
                   f'ND_Sum_{cur_gran}': ND_Sum, f'NRMSE_Sum_{cur_gran}': NRMSE_Sum})

    # test results for fine-grained dataset

    filename = f"{result_path}/output_{dataset_name}_{model_name}_{mg_dict}h_{cur_gran}h_{diff_steps}_{weights}_ratio{share_ratio_list}.csv"
    if not os.path.exists(filename):
        with open(filename, mode="a") as f:
            f.write("epoch,model_name,CRPS,ND,NRMSE,CRPS_Sum,ND_Sum,NRMSE_Sum\n")

    result_str = f"{epoch}, {model_name}, {CRPS}, {ND}, {NRMSE}, {CRPS_Sum}, {ND_Sum}, {NRMSE_Sum}\n"
    with open(filename, mode="a") as f:  # append the column names to the file
        f.write(result_str)
    plot(targets_list[cur_gran_index][0], forecasts_list[cur_gran_index][0], prediction_length=dataset.metadata.prediction_length,
         fname=f"{result_path}/plot_{dataset_name}_{model_name}_{mg_dict}h_{cur_gran}h_{diff_steps}_{weights}_ratio{share_ratio_list}.png")
