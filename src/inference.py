import random
import numpy as np
import pandas as pd
import os
import re
import json
import time
import copy
import scipy
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.metrics import r2_score
from asyncio import FastChildWatcher
from gvp_gnn import StructureEncoder
import data_transform_property, data_transform_pdi, data_transform_ppi
import model_property, model_pdi, model_ppi
import train_test_property, train_test_pdi, train_test_ppi
from visualization_train_dev_loss import plot_train_dev_metric



def launch(file_path, task_type):
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    
    max_pro_seq_len = 1022
    batch_size = 64
    add_structure = True
    add_goterm = True
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def find_application(value):
        categories = {
            "property": ["stability", "fluorescence", "remote homology", "secondary structure", "antigen binding"],
            "pdi": ["pdbbind", "kinase"],
            "ppi": ["skempi"]
        }
        for key, values in categories.items():
            values = [v.split() for v in values]
            values = sum(values, [])
            for v in values:
                if v in value:
                    return key
        return None
    
    application = find_application(task_type)

    if application == 'property':
        model = model_property.create_model(max_pro_seq_len, add_structure, add_goterm, task_type)
    elif application == 'pdi':
        model = model_pdi.create_model(max_pro_seq_len, add_structure, add_goterm, task_type)
    elif application == 'ppi':
        model = model_ppi.create_model(max_pro_seq_len, add_structure, add_goterm, task_type)
    else:
        raise ValueError("Unkown Application.")

    # Loading model checkpoint
    model_ckpt_dir = f"../checkpoints/{task_type}"
    os.makedirs(model_ckpt_dir, exist_ok=True)
    finetune_dict = torch.load(os.path.join(model_ckpt_dir, 'best-model.pth'), map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    finetune_dict = {k: v for k, v in finetune_dict.items() if k in model_dict}
    model_dict.update(finetune_dict)
    model.load_state_dict(model_dict)
    
    best_model = model.to(device)

    inference_dict = {}
    dir_input = os.path.dirname(file_path)
    if application == 'property':
        train_dataset_type = 'train'
        dev_dataset_type = 'valid'
        test_dataset_type = 'test'
        tester = train_test_property.Tester(best_model, batch_size, task_type)
        dataset_train = data_transform_property.data_read(dir_input, file_path, train_dataset_type, task_type, max_pro_seq_len, SEED, task_type)
        dataset_dev = data_transform_property.data_read(dir_input, file_path, dev_dataset_type, task_type, max_pro_seq_len, SEED, task_type)
        dataset_test = data_transform_property.data_read(dir_input, file_path, test_dataset_type, task_type, max_pro_seq_len, SEED, task_type)
        if len(dataset_train) != 0:
            dataset_train_pack, dataset_train_structure = data_transform_property.transform_data(dataset_train, max_pro_seq_len, dir_input, task_type)
            dataset_train_tuple = (dataset_train_pack, dataset_train_structure)
            loss_train, all_predict_labels_train, all_real_labels_train, all_pro_ids_train = tester.test(dataset_train_tuple, device, dir_input, train_dataset_type)
            inference_dict['train'] = list(zip(all_pro_ids_train, all_real_labels_train, all_predict_labels_train))
        if len(dataset_dev) != 0:
            dataset_dev_pack, dataset_dev_structure = data_transform_property.transform_data(dataset_dev, max_pro_seq_len, dir_input, task_type)
            dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure)
            loss_dev, all_predict_labels_dev, all_real_labels_dev, all_pro_ids_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_dataset_type)
            inference_dict['valid'] = list(zip(all_pro_ids_dev, all_real_labels_dev, all_predict_labels_dev))
        if len(dataset_test) != 0:
            dataset_test_pack, dataset_test_structure = data_transform_property.transform_data(dataset_test, max_pro_seq_len, dir_input, task_type)
            dataset_test_tuple = (dataset_test_pack, dataset_test_structure)
            loss_test, all_predict_labels_test, all_real_labels_test, all_pro_ids_test = tester.test(dataset_test_tuple, device, dir_input, test_dataset_type)
            inference_dict['test'] = list(zip(all_pro_ids_test, all_real_labels_test, all_predict_labels_test))
        
    elif application == 'pdi':
        train_dataset_type = 'train'
        dev_dataset_type = 'valid'
        test_dataset_type = 'test'
        tester = train_test_pdi.Tester(best_model, batch_size, task_type)
        dataset_train = data_transform_pdi.data_read(dir_input, file_path, train_dataset_type, max_pro_seq_len, SEED)
        dataset_dev = data_transform_pdi.data_read(dir_input, file_path, dev_dataset_type, max_pro_seq_len, SEED)
        dataset_test = data_transform_pdi.data_read(dir_input, file_path, test_dataset_type, max_pro_seq_len, SEED)
        if len(dataset_train) != 0:
            dataset_train_pack, dataset_train_structure = data_transform_pdi.transform_data(dataset_train, max_pro_seq_len, dir_input, train_dataset_type, task_type)
            dataset_train_tuple = (dataset_train_pack, dataset_train_structure)
            loss_train, all_predict_labels_train, all_real_labels_train, all_pro_ids_train = tester.test(dataset_train_tuple, device, dir_input, train_dataset_type)
            inference_dict['train'] = list(zip(all_pro_ids_train, all_real_labels_train, all_predict_labels_train))
        if len(dataset_dev) != 0:
            dataset_dev_pack, dataset_dev_structure = data_transform_pdi.transform_data(dataset_dev, max_pro_seq_len, dir_input, dev_dataset_type, task_type)
            dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure)
            loss_dev, all_predict_labels_dev, all_real_labels_dev, all_pro_ids_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_dataset_type)
            inference_dict['valid'] = list(zip(all_pro_ids_dev, all_real_labels_dev, all_predict_labels_dev))
        if len(dataset_test) != 0:
            dataset_test_pack, dataset_test_structure = data_transform_pdi.transform_data(dataset_test, max_pro_seq_len, dir_input, test_dataset_type, task_type)
            dataset_test_tuple = (dataset_test_pack, dataset_test_structure)
            loss_test, all_predict_labels_test, all_real_labels_test, all_pro_ids_test = tester.test(dataset_test_tuple, device, dir_input, test_dataset_type)
            inference_dict['test'] = list(zip(all_pro_ids_test, all_real_labels_test, all_predict_labels_test))

    elif application == 'ppi':
        tester = train_test_ppi.Tester(best_model, batch_size, task_type)
        if task_type == 'skempi':
            train_dataset_type = 'train'
            dev_dataset_type = 'valid'
            test_dataset_type = 'test'
            inference_dict['test'] = {}
            for fold_i in range(1, 11):
                # Loading model checkpoint
                model_ckpt_dir = f"../checkpoints/{task_type}/fold{fold_i+1}"
                os.makedirs(model_ckpt_dir, exist_ok=True)
                finetune_dict = torch.load(os.path.join(model_ckpt_dir, 'best-model.pth'), map_location=torch.device('cpu'))
                model_dict = model.state_dict()
                finetune_dict = {k: v for k, v in finetune_dict.items() if k in model_dict}
                model_dict.update(finetune_dict)
                model.load_state_dict(model_dict)
                best_model = model.to(device)
                dataset_test = data_transform_ppi.data_read(dir_input, file_path, f"fold{fold_i}-" + test_dataset_type, max_pro_seq_len, SEED)
                if len(dataset_test) != 0:
                    dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2 = data_transform_ppi.transform_data(dataset_test, max_pro_seq_len, dir_input)
                    dataset_test_tuple = (dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2)
                    loss_test, all_predict_labels_test, all_real_labels_test, all_pro_ids_test = tester.test(dataset_test_tuple, device, dir_input, 'all')
                    inference_dict['test'][f"fold{fold_i}"] = list(zip(all_pro_ids_test, all_real_labels_test, all_predict_labels_test))
        else:
            train_dataset_type = 'train'
            dev_dataset_type = 'valid'
            test_dataset_type = 'test'
            dataset_train = data_transform_ppi.data_read(dir_input, file_path, train_dataset_type, max_pro_seq_len, SEED)
            dataset_dev = data_transform_ppi.data_read(dir_input, file_path, dev_dataset_type, max_pro_seq_len, SEED)
            dataset_test = data_transform_ppi.data_read(dir_input, file_path, test_dataset_type, max_pro_seq_len, SEED)
            if len(dataset_train) != 0:
                dataset_train_pack, dataset_train_structure_1, dataset_train_structure_2 = data_transform_ppi.transform_data(dataset_train, max_pro_seq_len, dir_input)
                dataset_train_tuple = (dataset_train_pack, dataset_train_structure_1, dataset_train_structure_2)
                loss_train, all_predict_labels_train, all_real_labels_train, all_pro_ids_train = tester.test(dataset_train_tuple, device, dir_input, train_dataset_type)
                inference_dict['train'] = list(zip(all_pro_ids_train, all_real_labels_train, all_predict_labels_train))
            if len(dataset_dev) != 0:
                dataset_dev_pack, dataset_dev_structure_1, dataset_dev_structure_2 = data_transform_ppi.transform_data(dataset_dev, max_pro_seq_len, dir_input)
                dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure_1, dataset_dev_structure_2)
                loss_dev, all_predict_labels_dev, all_real_labels_dev, all_pro_ids_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_dataset_type)
                inference_dict['valid'] = list(zip(all_pro_ids_dev, all_real_labels_dev, all_predict_labels_dev))
            if len(dataset_test) != 0:
                dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2 = data_transform_ppi.transform_data(dataset_test, max_pro_seq_len, dir_input)
                dataset_test_tuple = (dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2)
                loss_test, all_predict_labels_test, all_real_labels_test, all_pro_ids_test = tester.test(dataset_test_tuple, device, dir_input, test_dataset_type)
                inference_dict['test'] = list(zip(all_pro_ids_test, all_real_labels_test, all_predict_labels_test))

    
    inference_results_dir = "inference_results"
    os.makedirs(inference_results_dir, exist_ok=True)
    
    with open(os.path.join(inference_results_dir, f"{task_type}_inference.json"), 'w') as f:
        json.dump(inference_dict, f, indent=4)
    f.close()
    
    return os.path.join(inference_results_dir, f"{task_type}_inference.json")