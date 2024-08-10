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
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
from asyncio import FastChildWatcher
from gvp_gnn import StructureEncoder
import data_transform_property, data_transform_pdi, data_transform_ppi
import model_property, model_pdi, model_ppi
import train_test_property, train_test_pdi, train_test_ppi
from visualization_train_dev_loss import plot_train_dev_metric


def finetune(model, dataset_tuple, dataset_type, dir_input, task_type, batch_size, epochs, model_ckpt_dir, device):
    assert len(dataset_tuple) == 2 and len(dataset_type) == 2
    dataset_train_tuple, dataset_dev_tuple = dataset_tuple
    train_dataset_type, dev_dataset_type = dataset_type
    lr = 1e-4
    lr_decay = 1.0
    weight_decay = 1e-4
    decay_interval = 5
    gradient_accumulation = 8
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    if task_type in ['stability', 'fluorescence'] or ('remote' in task_type or 'homology' in task_type) \
        or ('secondary' in task_type or 'structure' in task_type) or ('antigen' in task_type or 'binding' in task_type):
        trainer = train_test_property.Trainer(model, lr, weight_decay, batch_size, gradient_accumulation, task_type)
        tester = train_test_property.Tester(model, batch_size, task_type)
    elif task_type in ['pdbbind', 'kinase']:
        trainer = train_test_pdi.Trainer(model, lr, weight_decay, batch_size, gradient_accumulation, task_type)
        tester = train_test_pdi.Tester(model, batch_size, task_type)
    elif task_type in ['skempi']:
        trainer = train_test_ppi.Trainer(model, lr, weight_decay, batch_size, gradient_accumulation, task_type)
        tester = train_test_ppi.Tester(model, batch_size, task_type)
        
    min_loss_dev = float('inf')
    best_epoch = 0
    loss_train_epochs, loss_dev_epochs = [], []
    if task_type in ['stability', 'fluorescence', 'pdbbind']:
        spear_train_epochs, spear_dev_epochs = [], []
        rmse_train_epochs, rmse_dev_epochs = [], []
        mae_train_epochs, mae_dev_epochs = [], []
    elif ('remote' in task_type or 'homology' in task_type) or ('secondary' in task_type or 'structure' in task_type):
        acc_train_epochs, acc_dev_epochs = [], []
    elif task_type in ['kinase'] or ('antigen' in task_type or 'binding' in task_type):
        auc_train_epochs, auc_dev_epochs = [], []
        prc_train_epochs, prc_dev_epochs = [], []
    
    for epoch in range(1, epochs + 1):
        print('[Epoch: %d]' % epoch)
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        loss_train, all_predict_labels_train, all_real_labels_train, all_pro_ids_train = trainer.train(dataset_train_tuple, device, dir_input, train_dataset_type)
        loss_dev, all_predict_labels_dev, all_real_labels_dev, all_pro_ids_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_dataset_type)
        if task_type in ['stability', 'fluorescence', 'pdbbind']:
            spear_train, p_train = scipy.stats.spearmanr(all_real_labels_train, all_predict_labels_train)
            spear_dev, p_dev = scipy.stats.spearmanr(all_real_labels_dev, all_predict_labels_dev)
            rmse_train = train_test_property.get_rmse(all_real_labels_train, all_predict_labels_train)
            rmse_dev = train_test_property.get_rmse(all_real_labels_dev, all_predict_labels_dev)
            mae_train = train_test_property.get_mae(all_real_labels_train, all_predict_labels_train)
            mae_dev = train_test_property.get_mae(all_real_labels_dev, all_predict_labels_dev)
        elif 'remote' in task_type or 'homology' in task_type:
            acc_train = accuracy_score(all_real_labels_train, all_predict_labels_train)
            acc_dev = accuracy_score(all_real_labels_dev, all_predict_labels_dev)
        elif 'secondary' in task_type or 'structure' in task_type:
            acc_train = accuracy_score(sum(all_real_labels_train, []), sum(all_predict_labels_train, []))
            acc_dev = accuracy_score(sum(all_real_labels_dev, []), sum(all_predict_labels_dev, []))
        elif task_type in ['kinase'] or ('antigen' in task_type or 'binding' in task_type):
            auc_train = roc_auc_score(all_real_labels_train, all_predict_labels_train)
            auc_dev = roc_auc_score(all_real_labels_dev, all_predict_labels_dev)
            tpr_train, fpr_train, _ = precision_recall_curve(all_real_labels_train, all_predict_labels_train)
            prc_train = auc(fpr_train, tpr_train)
            tpr_dev, fpr_dev, _ = precision_recall_curve(all_real_labels_dev, all_predict_labels_dev)
            prc_dev = auc(fpr_dev, tpr_dev)
            
        loss_train_epochs.append(float("%.3f" % loss_train)), loss_dev_epochs.append(float("%.3f" % loss_dev))
        if task_type in ['stability', 'fluorescence', 'pdbbind']:
            spear_train_epochs.append(float("%.3f" % spear_train)), spear_dev_epochs.append(float("%.3f" % spear_dev))
            rmse_train_epochs.append(float("%.3f" % rmse_train)), rmse_dev_epochs.append(float("%.3f" % rmse_dev))
            mae_train_epochs.append(float("%.3f" % mae_train)), mae_dev_epochs.append(float("%.3f" % mae_dev))
        elif ('remote' in task_type or 'homology' in task_type) or ('secondary' in task_type or 'structure' in task_type):
            acc_train_epochs.append(float("%.3f" % acc_train)), acc_dev_epochs.append(float("%.3f" % acc_dev))
        elif task_type in ['kinase'] or ('antigen' in task_type or 'binding' in task_type):
            auc_train_epochs.append(float("%.3f" % auc_train)), auc_dev_epochs.append(float("%.3f" % auc_dev))
            prc_train_epochs.append(float("%.3f" % prc_train)), prc_dev_epochs.append(float("%.3f" % prc_dev))
            
        if loss_dev < min_loss_dev:
            tester.save_model(model, os.path.join(model_ckpt_dir, 'model-best.pth'))
            best_model = copy.deepcopy(model)
            min_loss_dev = loss_dev
            best_epoch = epoch

    dict_loss = {}
    dict_loss['epochs'] = list(range(1, epochs+1))
    dict_loss['loss_train'] = loss_train_epochs
    dict_loss['loss_dev'] = loss_dev_epochs
    if task_type in ['stability', 'fluorescence', 'pdbbind']:
        dict_loss['spear_train'] = spear_train_epochs
        dict_loss['spear_dev'] = spear_dev_epochs
        dict_loss['rmse_train'] = rmse_train_epochs
        dict_loss['rmse_dev'] = rmse_dev_epochs
        dict_loss['mae_train'] = mae_train_epochs
        dict_loss['mae_dev'] = mae_dev_epochs
    elif ('remote' in task_type or 'homology' in task_type) or ('secondary' in task_type or 'structure' in task_type):
        dict_loss['acc_train'] = acc_train_epochs
        dict_loss['acc_dev'] = acc_dev_epochs
    elif task_type in ['kinase'] or ('antigen' in task_type or 'binding' in task_type):
        dict_loss['auc_train'] = auc_train_epochs
        dict_loss['auc_dev'] = auc_dev_epochs
        dict_loss['prc_train'] = prc_train_epochs
        dict_loss['prc_dev'] = prc_dev_epochs
    df_loss = pd.DataFrame(dict_loss)
    df_loss.to_csv(os.path.join(model_ckpt_dir, f"{task_type}_log.csv"), index=False)

    if task_type in ['stability', 'fluorescence', 'pdbbind']:
        plot_train_dev_metric(list(range(1, epochs+1)), loss_train_epochs, loss_dev_epochs, model_ckpt_dir, 'Loss', task_type)
        plot_train_dev_metric(list(range(1, epochs+1)), spear_train_epochs, spear_dev_epochs, model_ckpt_dir, 'Spearman', task_type)
        plot_train_dev_metric(list(range(1, epochs+1)), rmse_train_epochs, rmse_dev_epochs, model_ckpt_dir, 'RMSE', task_type)
        plot_train_dev_metric(list(range(1, epochs+1)), mae_train_epochs, mae_dev_epochs, model_ckpt_dir, 'MAE', task_type)
    elif ('remote' in task_type or 'homology' in task_type) or ('secondary' in task_type or 'structure' in task_type):
        plot_train_dev_metric(list(range(1, epochs+1)), loss_train_epochs, loss_dev_epochs, model_ckpt_dir, 'Loss', task_type)
        plot_train_dev_metric(list(range(1, epochs+1)), acc_train_epochs, acc_dev_epochs, model_ckpt_dir, 'Accuracy', task_type)
    elif task_type in ['kinase'] or ('antigen' in task_type or 'binding' in task_type):
        plot_train_dev_metric(list(range(1, epochs+1)), loss_train_epochs, loss_dev_epochs, model_ckpt_dir, 'Loss', task_type)
        plot_train_dev_metric(list(range(1, epochs+1)), auc_train_epochs, auc_dev_epochs, model_ckpt_dir, 'AUC', task_type)
        plot_train_dev_metric(list(range(1, epochs+1)), prc_train_epochs, prc_dev_epochs, model_ckpt_dir, 'PRC', task_type)

    return best_model

    

def launch(file_path, task_type):
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    max_pro_seq_len = 1022
    batch_size = 4
    epochs = 50
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

    model_ckpt_dir = f"../checkpoints/{task_type}"
    os.makedirs(model_ckpt_dir, exist_ok=True)
    
    inference_dict = {}
    dir_input = os.path.dirname(file_path)
    if application == 'property':
        train_dataset_type = 'train'
        dev_dataset_type = 'valid'
        test_dataset_type = 'test'
        dataset_train = data_transform_property.data_read(dir_input, file_path, train_dataset_type, task_type, max_pro_seq_len, SEED, task_type)
        dataset_dev = data_transform_property.data_read(dir_input, file_path, dev_dataset_type, task_type, max_pro_seq_len, SEED, task_type)
        dataset_test = data_transform_property.data_read(dir_input, file_path, test_dataset_type, task_type, max_pro_seq_len, SEED, task_type)
        if len(dataset_dev) == 0:
            dev_dataset_type = train_dataset_type
            dataset_train, dataset_dev = dataset_train[:int(len(dataset_train) * 0.7)], dataset_train[int(len(dataset_train) * 0.7):]
        dataset_train_pack, dataset_train_structure = data_transform_property.transform_data(dataset_train, max_pro_seq_len, dir_input, task_type)
        dataset_train_tuple = (dataset_train_pack, dataset_train_structure)
        dataset_dev_pack, dataset_dev_structure = data_transform_property.transform_data(dataset_dev, max_pro_seq_len, dir_input, task_type)
        dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure)
        dataset_test_pack, dataset_test_structure = data_transform_property.transform_data(dataset_test, max_pro_seq_len, dir_input, task_type)
        dataset_test_tuple = (dataset_test_pack, dataset_test_structure)

        best_model = finetune(model, (dataset_train_tuple, dataset_dev_tuple), (train_dataset_type, dev_dataset_type), dir_input, task_type, batch_size, epochs, model_ckpt_dir, device)
        tester = train_test_property.Tester(best_model, batch_size, task_type)
        
        loss_train, all_predict_labels_train, all_real_labels_train, all_pro_ids_train = tester.test(dataset_train_tuple, device, dir_input, train_dataset_type)
        inference_dict['train'] = list(zip(all_pro_ids_train, all_real_labels_train, all_predict_labels_train))
        loss_dev, all_predict_labels_dev, all_real_labels_dev, all_pro_ids_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_dataset_type)
        inference_dict['valid'] = list(zip(all_pro_ids_dev, all_real_labels_dev, all_predict_labels_dev))
        loss_test, all_predict_labels_test, all_real_labels_test, all_pro_ids_test = tester.test(dataset_test_tuple, device, dir_input, test_dataset_type)
        inference_dict['test'] = list(zip(all_pro_ids_test, all_real_labels_test, all_predict_labels_test))

    elif application == 'pdi':
        train_dataset_type = 'train'
        dev_dataset_type = 'valid'
        test_dataset_type = 'test'
        dataset_train = data_transform_pdi.data_read(dir_input, file_path, train_dataset_type, max_pro_seq_len, SEED)
        dataset_dev = data_transform_pdi.data_read(dir_input, file_path, dev_dataset_type, max_pro_seq_len, SEED)
        dataset_test = data_transform_pdi.data_read(dir_input, file_path, test_dataset_type, max_pro_seq_len, SEED)
        if len(dataset_dev) == 0:
            dev_dataset_type = train_dataset_type
            dataset_train, dataset_dev = dataset_train[:int(len(dataset_train) * 0.7)], dataset_train[int(len(dataset_train) * 0.7):]
        dataset_train_pack, dataset_train_structure = data_transform_pdi.transform_data(dataset_train, max_pro_seq_len, dir_input, train_dataset_type, task_type)
        dataset_train_tuple = (dataset_train_pack, dataset_train_structure)
        dataset_dev_pack, dataset_dev_structure = data_transform_pdi.transform_data(dataset_dev, max_pro_seq_len, dir_input, dev_dataset_type, task_type)
        dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure)
        dataset_test_pack, dataset_test_structure = data_transform_pdi.transform_data(dataset_test, max_pro_seq_len, dir_input, test_dataset_type, task_type)
        dataset_test_tuple = (dataset_test_pack, dataset_test_structure)

        best_model = finetune(model, (dataset_train_tuple, dataset_dev_tuple), (train_dataset_type, dev_dataset_type), dir_input, task_type, batch_size, epochs, model_ckpt_dir, device)
        tester = train_test_pdi.Tester(best_model, batch_size, task_type)
        
        loss_train, all_predict_labels_train, all_real_labels_train, all_pro_ids_train = tester.test(dataset_train_tuple, device, dir_input, train_dataset_type)
        inference_dict['train'] = list(zip(all_pro_ids_train, all_real_labels_train, all_predict_labels_train))
        loss_dev, all_predict_labels_dev, all_real_labels_dev, all_pro_ids_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_dataset_type)
        inference_dict['valid'] = list(zip(all_pro_ids_dev, all_real_labels_dev, all_predict_labels_dev))
        loss_test, all_predict_labels_test, all_real_labels_test, all_pro_ids_test = tester.test(dataset_test_tuple, device, dir_input, test_dataset_type)
        inference_dict['test'] = list(zip(all_pro_ids_test, all_real_labels_test, all_predict_labels_test))

    elif application == 'ppi':
        if task_type == 'skempi':
            train_dataset_type = 'train'
            dev_dataset_type = 'valid'
            test_dataset_type = 'test'
            inference_dict['train'] = {}
            inference_dict['test'] = {}
            for fold_i in range(1, 11):
                print(f'[Fold{fold_i+1}]')
                os.makedirs(os.path.join(model_ckpt_dir, f'fold{fold_i}'), exist_ok=True)
                dataset_train = data_transform_ppi.data_read(dir_input, file_path, f"fold{fold_i}-" + train_dataset_type, max_pro_seq_len, SEED)
                dataset_test = data_transform_ppi.data_read(dir_input, file_path, f"fold{fold_i}-" + test_dataset_type, max_pro_seq_len, SEED)
                
                dataset_train_pack, dataset_train_structure_1, dataset_train_structure_2 = data_transform_ppi.transform_data(dataset_train, max_pro_seq_len, dir_input)
                dataset_train_tuple = (dataset_train_pack, dataset_train_structure_1, dataset_train_structure_2)
                dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2 = data_transform_ppi.transform_data(dataset_test, max_pro_seq_len, dir_input)
                dataset_test_tuple = (dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2)
                
                best_model = finetune(model, (dataset_train_tuple, dataset_test_tuple), ('all', 'all'), dir_input, task_type, batch_size, epochs, os.path.join(model_ckpt_dir, f'fold{fold_i}'), device)
                tester = train_test_ppi.Tester(best_model, batch_size, task_type)
            
                loss_train, all_predict_labels_train, all_real_labels_train, all_pro_ids_train = tester.test(dataset_train_tuple, device, dir_input, 'all')
                inference_dict['train'][f"fold{fold_i}"] = list(zip(all_pro_ids_train, all_real_labels_train, all_predict_labels_train))         
                loss_test, all_predict_labels_test, all_real_labels_test, all_pro_ids_test = tester.test(dataset_test_tuple, device, dir_input, 'all')
                inference_dict['test'][f"fold{fold_i}"] = list(zip(all_pro_ids_test, all_real_labels_test, all_predict_labels_test))
        else:
            train_dataset_type = 'train'
            dev_dataset_type = 'valid'
            test_dataset_type = 'test'
            dataset_train = data_transform_ppi.data_read(dir_input, file_path, train_dataset_type, max_pro_seq_len, SEED)
            dataset_dev = data_transform_ppi.data_read(dir_input, file_path, dev_dataset_type, max_pro_seq_len, SEED)
            dataset_test = data_transform_ppi.data_read(dir_input, file_path, test_dataset_type, max_pro_seq_len, SEED)
            if len(dataset_dev) == 0:
                dev_dataset_type = train_dataset_type
                dataset_train, dataset_dev = dataset_train[:int(len(dataset_train) * 0.7)], dataset_train[int(len(dataset_train) * 0.7):]
            dataset_train_pack, dataset_train_structure_1, dataset_train_structure_2 = data_transform_ppi.transform_data(dataset_train, max_pro_seq_len, dir_input)
            dataset_train_tuple = (dataset_train_pack, dataset_train_structure_1, dataset_train_structure_2)
            dataset_dev_pack, dataset_dev_structure_1, dataset_dev_structure_2 = data_transform_ppi.transform_data(dataset_dev, max_pro_seq_len, dir_input)
            dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure_1, dataset_dev_structure_2)
            dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2 = data_transform_ppi.transform_data(dataset_test, max_pro_seq_len, dir_input)
            dataset_test_tuple = (dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2)

            best_model = finetune(model, (dataset_train_tuple, dataset_dev_tuple), (train_dataset_type, dev_dataset_type), dir_input, task_type, batch_size, epochs, model_ckpt_dir, device)
            tester = train_test_pdi.Tester(best_model, batch_size, task_type)

            loss_train, all_predict_labels_train, all_real_labels_train, all_pro_ids_train = tester.test(dataset_train_tuple, device, dir_input, train_dataset_type)
            inference_dict['train'] = list(zip(all_pro_ids_train, all_real_labels_train, all_predict_labels_train))
            loss_dev, all_predict_labels_dev, all_real_labels_dev, all_pro_ids_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_dataset_type)
            inference_dict['valid'] = list(zip(all_pro_ids_dev, all_real_labels_dev, all_predict_labels_dev))
            loss_test, all_predict_labels_test, all_real_labels_test, all_pro_ids_test = tester.test(dataset_test_tuple, device, dir_input, test_dataset_type)
            inference_dict['test'] = list(zip(all_pro_ids_test, all_real_labels_test, all_predict_labels_test))

    inference_results_dir = "inference_results"
    os.makedirs(inference_results_dir, exist_ok=True)
    
    with open(os.path.join(inference_results_dir, f"{task_type}_inference.json"), 'w') as f:
        json.dump(inference_dict, f, indent=4)
    f.close()
    
    return os.path.join(inference_results_dir, f"{task_type}_inference.json")