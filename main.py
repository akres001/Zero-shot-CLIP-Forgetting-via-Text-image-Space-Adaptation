import os
# import torch.optim as optim 
import numpy as np
import json
import random
from tqdm import tqdm
import torch
import pickle
import argparse

import datasets.stanford_cars
import datasets.stanford_dogs
import datasets.caltech101
import datasets.oxford_flowers
import datasets.oxford_pets
import datasets.food101
import datasets.pinsfaces

from dassl.data.datasets.build import DATASET_REGISTRY, build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from torch.utils.data import DataLoader, Dataset
from dassl.config import get_cfg_default

from clip import clip

from utils import *
import utils

from utils_lora import *
from collections import OrderedDict

from gen_classes import *
from forget_cls import *

torch.set_num_threads(10)

device = 'cuda'
IGNORE_OTHER_DS = False
PRINT_EVERY = 200
EPOCHS = 2000
REDUCTION_THR = 0.7
UNLEARN_TRIALS = 100

CUSTOM_TEMPLATES = {
        "OxfordFlowers": "a photo of a {}, a type of flower.",
        "StanfordCars": "a photo of a {}.",
        "Caltech101": "a photo of a {}.",
        "StanfordDogs": "a photo of a {}.",
        }
        
def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def initialize_config(args):
    cfg = get_cfg_default()
    cfg.merge_from_file(args.config_file)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.SEED = args.seed
    cfg.DATASET.ROOT = "/app/datasets/"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASET.NUM_SHOTS = -1
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 4
    cfg.DATALOADER.TEST.BATCH_SIZE = 128
    return cfg


def load_test_datasets(cfg):
    test_datasets, test_dataloaders, datasets_cls = {}, {}, {}
    for ds in all_ds:
        cfg.DATASET.NAME = ds
        tfm_train = build_transform(cfg, is_train=True)
        tfm_test = build_transform(cfg, is_train=False)
        dataset = build_dataset(cfg)

        test_loader = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.test,
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    dataset_wrapper=None
        )

        test_datasets[ds] = dataset
        test_dataloaders[ds] = test_loader
        datasets_cls[ds] = dataset

    return test_datasets, test_dataloaders, datasets_cls

    
def get_activation_proj(name):
    def hook(model, input, output):
        if 'ln_final' in name:
            output = output[0]
            hooks[name] = (input[0].detach(), output)
    return hook

hooks = {}

@torch.no_grad()
def precompute_projections(model, classes, template=['{}']):
    list_hooks = []
    projections = []
    global hooks
    
    for classname in classes:
        hooks = {}
        classname = classname.replace('_', ' ')
        texts = [t.format(classname) for t in template]
        texts = clip.tokenize(texts).to(model.visual.conv1.weight.device)
        class_embeddings = model.encode_text(texts)
        projections.append(class_embeddings)
        list_hooks.append(hooks['ln_final'][1][texts.argmax(-1)].clone())
        
    projections = torch.stack(projections, dim=1)
    list_hooks = torch.cat(list_hooks)
    return projections, list_hooks


@torch.no_grad()
def register_model_hooks(model):
    for name, module in model.named_modules():
        module._forward_hooks = OrderedDict()
        
    for name, module in model.named_modules():
        module.register_forward_hook(get_activation_proj(name))
        
            
def get_preserved_classes(main_ds, forget_label, args, class_lists):
    classes_preserved_list = class_lists.get(main_ds, [])
    if args.multiclass_forget:
        preserved = set([cl.lower() for cl in classes_preserved_list]) - set([cl.lower() for cl in forget_label.split('|')])
    else:
        preserved = set([cl.lower() for cl in classes_preserved_list]) - {forget_label.lower()}
    return list(preserved)


@torch.no_grad()
def compute_proj_into(model, idx_cls_forget, idx_cls_forget_original, original_projections_norm, device):
    
    empty_text = clip.tokenize("").to(device)
    embed = model.encode_text(empty_text).repeat(idx_cls_forget.shape[0], 1)
    proj_into = embed / embed.norm(dim=-1, keepdim=True)

    while not ((idx_cls_forget_original.to(device) == (original_projections_norm @ proj_into.T).argmax(0)).sum() == 0):
        embed = model.encode_text(clip.tokenize("").to(device)).repeat(idx_cls_forget_original.shape[0], 1)
        embed += torch.randn(embed.size()).to(device) * 0.5
        proj_into = embed / embed.norm(dim=-1, keepdim=True)
        print(f"{100 * '*'} regenerating empty for emptytoken")

    return proj_into

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/result0", help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="only positive value enables a fixed seed")
    parser.add_argument("--run_ds", type=str, default="")
    parser.add_argument("--backbone_arch", type=str, default="RN50")
    parser.add_argument("--multiclass_forget", type=int, default=0)
    
    args = parser.parse_args()
    args.config_file = "configs/trainers/adam_lr2e-4_B256_ep200_ViT16.yaml"
    print("Arguments : ", args)
    
    set_seeds(args.seed)
    cfg = initialize_config(args)
    test_datasets, test_dataloaders, datasets_cls = load_test_datasets(cfg)
    os.makedirs(args.output_dir, exist_ok=True)   
    configs = get_configs(args)
        
    results_zs = load_results(args.backbone_arch)
    utils.CUSTOM_TEMPLATES = CUSTOM_TEMPLATES
    all_logs = {}
    
    if args.run_ds != "" :
        args.run_ds = [item for item in args.run_ds.split(',')]
        print(args.run_ds)
        assert all([ds in all_ds for ds in args.run_ds])
        run_ds = args.run_ds
    else:
        run_ds = all_ds[:]
    
    output_base = args.output_dir
    
    indices_by_ds_all = {}
    last_idx = 0
    for ds in all_ds:
        indices_by_ds_all[ds] = torch.arange(last_idx, last_idx + len(datasets_cls[ds].classnames))
        last_idx += torch.arange(len(datasets_cls[ds].classnames)).shape[0]
        
    model = get_model(device=device, arch=args.backbone_arch) 
    register_model_hooks(model)
                
    # precompute projections for all datasets and classes
    original_projections_all = []
    final_hooks_all = []
    all_ds_classes = []

    for main_ds in all_ds:
        print(main_ds)
        ds_classes = datasets_cls[main_ds].classnames
        projections, list_hooks = precompute_projections(model, ds_classes)

        final_hooks_all.append(list_hooks)
        original_projections_all.append(projections[0])

        all_ds_classes.extend(ds_classes)
        
    final_hooks_all = torch.cat(final_hooks_all)
    original_projections_all = torch.cat(original_projections_all)
    
    for main_ds in run_ds:
        
        kwargs = {
              'lamb_preserve': configs[main_ds]['lamb_preserve'],
              'lamb_forget': configs[main_ds]['lamb_forget'],
              'lora_r': configs[main_ds]['lora_r'],
              'lamb_weight': configs[main_ds]['lamb_weight'],
          }
       
        args.output_dir = output_base + f"/{main_ds}"
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_base + "/args.txt", "w") as f:
            f.write(str(args)) 
        
        other_ds = all_ds[:]
        other_ds.remove(main_ds)
        all_logs[main_ds] = {'settings': {'kwargs': kwargs}}
        
        forget_classes_list = [forget_classes_all[main_ds]] if args.multiclass_forget else forget_classes_all[main_ds]
        
        print("KWARGS", kwargs)
        for forget_label in forget_classes_list:
                        
            idx_cls_forget = []
            idx_cls_forget_original = []
            if args.multiclass_forget:
                for c in forget_label:
                    idx_cls_forget.append(all_ds_classes.index(c))
                    idx_cls_forget_original.append(datasets_cls[main_ds].classnames.index(c))
                forget_label = '|'.join(forget_label)
            else:
                idx_cls_forget.append(all_ds_classes.index(forget_label))
                idx_cls_forget_original.append(datasets_cls[main_ds].classnames.index(forget_label))
                
            idx_cls_forget = torch.tensor(idx_cls_forget).long()
            idx_cls_forget_original = torch.tensor(idx_cls_forget_original).long()
            
            original_projections_all_full = original_projections_all.clone()
            change_hooks = final_hooks_all[idx_cls_forget, :]
            original_projections_norm = original_projections_all_full[indices_by_ds_all[main_ds]] / original_projections_all_full[indices_by_ds_all[main_ds]].norm(dim=1, keepdim=True)

            model = get_model(device=device, arch=args.backbone_arch)
            register_model_hooks(model)
            
            classes_preserved_list = get_preserved_classes(main_ds, forget_label, args, {
                'OxfordFlowers': oxfordflowers_list,
                'StanfordDogs': stanforddogs_list,
                'StanfordCars': stanfordcars_list,
                'Caltech101': caltech_list
            })
            
            projection_additional, preserve_hooks = precompute_projections(model, classes_preserved_list)
            preserve_output = projection_additional[0]
            
            all_logs[main_ds][forget_label] = {}
            new_weights = {}
            r, lamb_preserve, lamb_forget, lamb_weight = kwargs['lora_r'], kwargs['lamb_preserve'],  kwargs['lamb_forget'],  kwargs['lamb_weight']
            
            print(100*"=")
            model = get_model(device=device, arch=args.backbone_arch)
            in_proj, out_proj = model.text_projection.shape
            
            proj_into = compute_proj_into(model, idx_cls_forget, idx_cls_forget_original, original_projections_norm, device)

                        
            for _ in range(UNLEARN_TRIALS):
                                
                new_text_proj = Linear(in_proj, out_proj, r=r, bias=False, device=device)
                new_text_proj.weight = torch.nn.Parameter(model.text_projection.T)
                new_text_proj.weight.requires_grad = False
                
                optimizer = torch.optim.Adam(list(new_text_proj.parameters()), lr=0.01)
                
                with torch.no_grad():
                    initial_forget_loss = torch.norm(proj_into - new_text_proj(change_hooks), p=2)
                
                best_loss = np.inf                
                for epoch in range(EPOCHS):
                    new_preserve_output = new_text_proj(preserve_hooks)
                    new_forget_output = new_text_proj(change_hooks)
                                        
                    weight_loss = torch.norm((new_text_proj.lora_B @ new_text_proj.lora_A).transpose(0, 1), p=2) 
                                            
                    loss = lamb_forget * torch.norm(proj_into - new_forget_output, p=2) + \
                           lamb_preserve * torch.norm(preserve_output - new_preserve_output, p=2) + \
                           lamb_weight * weight_loss

                    if epoch % PRINT_EVERY == 0:
                        print(torch.norm(proj_into - new_forget_output, p=2), torch.norm(preserve_output - new_preserve_output, p=2), weight_loss)

                    if loss < best_loss:
                        new_text_proj.train(False)
                        best_weights = new_text_proj.weight.clone()
                        new_text_proj.train(True)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                
                with torch.no_grad():
                    # make sure there was enough reduction
                    print("initial_forget_loss", initial_forget_loss)
                    final_proj = change_hooks @ best_weights.T
                    reduction = (initial_forget_loss - torch.norm(proj_into - final_proj, p=2)).abs() / initial_forget_loss
                    print("final", torch.norm(proj_into - final_proj, p=2))
                    if reduction < REDUCTION_THR:
                        lamb_forget += 0.05
                        print("New forget ", lamb_forget)
                        continue
                    else:
                        break
                
            new_weights[f"proj_weight"] = torch.nn.Parameter(best_weights)
            
            model.load_state_dict({**model.state_dict(), 'text_projection': new_weights[f"proj_weight"].T})
            torch.save(model.state_dict(), args.output_dir + f"/model_{main_ds}_{forget_label}.pth")            
            model.eval()
    
            if args.multiclass_forget:
                results_ds = eval_all_ds(model, datasets_cls, main_ds, forget_label, test_dataloaders, None, 
                     eval_forgetonly=IGNORE_OTHER_DS, debug=True, device=device, ignore_labels_main=forget_classes_list[0])
            else:
                results_ds = eval_all_ds(model, datasets_cls, main_ds, forget_label, test_dataloaders, 
                                         None, eval_forgetonly=IGNORE_OTHER_DS, debug=True, device=device)

            all_logs[main_ds][forget_label]['final_results'] = results_ds
            print(f"{20 * '*'} Final results for {forget_label}", results_ds[main_ds][forget_label])
            
            with open(output_base + "/logs.json", "w") as f:
                json.dump(all_logs, f)
            
        print(all_logs)


    with open(output_base + "/logs.json", "w") as f:
        json.dump(all_logs, f)