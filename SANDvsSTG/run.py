from stg import STG
import numpy as np
import torch
import os
import random
import json
import tensorflow as tf
from datasets.dataset import get_dataset
import hyperparams

def tf_to_numpy(ds):
    X_list, y_list = [], []
    for X_batch, y_batch in ds:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).argmax(axis=1)
    return X, y


for dataset_name in ['madelon']:
        for seed in range(1, 11):
            lmbda = hyperparams.LAMBDA[dataset_name]
            print(f"Running {dataset_name} with seed {seed} and lambda {lmbda}")
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            torch.manual_seed(seed)

            datasets = get_dataset(dataset_name, val_ratio=0.125, batch_size=hyperparams.BATCH[dataset_name])
            ds_train = datasets["ds_train"]
            ds_val = datasets["ds_val"]
            ds_test = datasets["ds_test"]
            X_train, y_train = tf_to_numpy(ds_train)
            X_val, y_val = tf_to_numpy(ds_val)
            X_test, y_test = tf_to_numpy(ds_test)

            model = STG(task_type='classification',input_dim=X_train.shape[1], output_dim=ds_train.element_spec[1].shape[1], hidden_dims=[hyperparams.DEEP_LAYERS[dataset_name]], activation='relu', optimizer='Adam', learning_rate=hyperparams.LEARNING_RATE[dataset_name], batch_size=hyperparams.BATCH[dataset_name], feature_selection=True, sigma=0.5, lam=lmbda, device="cpu") 
            model.fit(X_train, y_train, nr_epochs=hyperparams.EPOCHS[dataset_name], valid_X=X_val, valid_y=y_val, print_interval=9999)
            
            y_pred=model.predict(X_test)
            acc = np.mean(y_pred == y_test)

            gates_prob = model.get_gates(mode='prob')
            gates_raw = model.get_gates(mode='raw')

            os.makedirs(f"results/{dataset_name}/lambda{lmbda}", exist_ok=True)
            results = {
                "lambda": lmbda,
                "seed": seed,
                "accuracy": float(acc),
                "gates_prob": gates_prob.tolist() if isinstance(gates_prob, np.ndarray) else gates_prob,
                "gates_raw": gates_raw.tolist() if isinstance(gates_raw, np.ndarray) else gates_raw,
            }

            with open(f"results/{dataset_name}/lambda{lmbda}/seed{seed}.json", "w") as f:
                json.dump(results, f, indent=4)