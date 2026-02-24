import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import h5py

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

# peft is imported lazily inside load_clip() when use_lora=True

class CXRDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='report', size=None, transform=None):
        super().__init__()
        if size != None: 
            self.img_dset = h5py.File(img_path, 'r')['cxr'][:size]
            self.txt_dset = pd.read_csv(txt_path)[column][:size]
        else: 
            self.img_dset = h5py.File(img_path, 'r')['cxr']
            self.txt_dset = pd.read_csv(txt_path)[column]
        self.transform = transform
            
    def __len__(self):
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        txt = self.txt_dset[idx] # python str
        if type(txt) == type(float("nan")): # capture the case of empty "Impression" sections
            txt = " "

        img = torch.from_numpy(img) # torch, (3, 320, 320)
        if self.transform:
            img = self.transform(img)
        sample = {'img': img, 'txt': txt }
        
        return sample

def load_data(cxr_filepath, txt_filepath, batch_size=4, column='report', pretrained=False, use_dinov3=False, verbose=False):
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    if pretrained or use_dinov3:
        input_resolution = 448
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        print('Interpolation Mode: ', InterpolationMode.BICUBIC)
        if use_dinov3:
            print("Finished image transforms for DINOv3 model.")
        else:
            print("Finished image transforms for pretrained model.")
    else:
        input_resolution = 320
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])
        print("Finished image transforms for clip model.")
    
    torch_dset = CXRDataset(img_path=cxr_filepath,
                        txt_path=txt_filepath, column=column, transform=transform)
    
    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            plt.imshow(sample['img'][0])
            plt.show()
            print(i, sample['img'].size(), sample['txt'])
            if i == 3:
                break
    
    loader_params = {'batch_size':batch_size, 'shuffle': True, 'num_workers': 0}
    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device
    
def load_clip(model_path=None, pretrained=False, context_length=77,
              use_dinov3=False, dinov3_model_name="dinov3_vitb16",
              dinov3_repo_dir="/cbica/projects/CXR/codes/dinov3",
              dinov3_weights="/cbica/projects/CXR/codes/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
              freeze_dinov3=False,
              use_lora=False, lora_rank=16, lora_alpha=32, lora_dropout=0.05,
              lora_target_modules=None):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model
    architecture.

    args:
        * model_path (optional) - path to model weights that the model
        will be initialized with
        * pretrained (optional) - if True, will load the pretrained
        CLIP model
        * context_length (optional) - length of the maximum number of
        tokens that can be inputted into the CLIP model
        * use_dinov3 (optional) - if True, will use DINOv3 as vision encoder
        * dinov3_model_name (optional) - DINOv3 model variant to use
        * dinov3_repo_dir (optional) - path to local DINOv3 repo
        * dinov3_weights (optional) - path to DINOv3 pretrained weights
        * freeze_dinov3 (optional) - if True, freeze DINOv3 backbone
        * use_lora (optional) - if True, apply LoRA adapters to DINOv3 backbone
        * lora_rank (optional) - LoRA rank
        * lora_alpha (optional) - LoRA alpha scaling factor
        * lora_dropout (optional) - LoRA dropout rate
        * lora_target_modules (optional) - list of module names to apply LoRA to;
          if None, auto-detects attention linear layers
    '''

    params = {
        'embed_dim':768,
        'image_resolution': 320,
        'vision_layers': 12,
        'vision_width': 768,
        'vision_patch_size': 16,
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if pretrained and not use_dinov3:
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else:
        model = CLIP(**params)

        # Replace visual encoder with DINOv3 if requested
        if use_dinov3:
            # Enforce LoRA for vit7b16 (too large for full finetune)
            if dinov3_model_name == 'dinov3_vit7b16' and not use_lora:
                raise ValueError(
                    "dinov3_vit7b16 (6.7B params) is too large for full finetuning. "
                    "Please enable LoRA with --use_lora."
                )

            # Load DINOv3 backbone from local repo
            dinov3_backbone = torch.hub.load(
                dinov3_repo_dir, dinov3_model_name,
                source='local', weights=dinov3_weights
            )
            dinov3_backbone = dinov3_backbone.to(device)

            # Apply LoRA if requested
            if use_lora:
                from peft import LoraConfig, get_peft_model

                if freeze_dinov3:
                    print("Warning: --freeze_dinov3 is ignored when --use_lora is enabled "
                          "(LoRA already freezes base parameters).")

                # Auto-detect target modules if not specified
                if lora_target_modules is None:
                    detected = []
                    for name, module in dinov3_backbone.named_modules():
                        if isinstance(module, nn.Linear):
                            # Target attention + MLP linear layers (fc1/fc2 for standard MLP, w1/w2/w3 for SwiGLU)
                            short = name.split('.')[-1]
                            if short in ('qkv', 'proj', 'fc1', 'fc2', 'w1', 'w2', 'w3'):
                                detected.append(name)
                    # Deduplicate while preserving order
                    seen = set()
                    lora_target_modules = []
                    for n in detected:
                        if n not in seen:
                            seen.add(n)
                            lora_target_modules.append(n)
                    print(f"[LoRA] Auto-detected target modules ({len(lora_target_modules)}): "
                          f"{lora_target_modules[:10]}{'...' if len(lora_target_modules) > 10 else ''}")

                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=lora_target_modules,
                    bias="none",
                )
                dinov3_backbone = get_peft_model(dinov3_backbone, lora_config)

                # Print trainable vs total param counts
                trainable = sum(p.numel() for p in dinov3_backbone.parameters() if p.requires_grad)
                total = sum(p.numel() for p in dinov3_backbone.parameters())
                print(f"[LoRA] Trainable params: {trainable:,} / {total:,} "
                      f"({100 * trainable / total:.2f}%)")

            # Get feature dimension using a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                features = dinov3_backbone(dummy_input)
                backbone_dim = features.shape[-1]

            # Create simple wrapper class
            class DINOv3Visual(nn.Module):
                def __init__(self, backbone, backbone_dim, output_dim):
                    super().__init__()
                    self.backbone = backbone
                    self.projection = nn.Linear(backbone_dim, output_dim)

                def forward(self, x):
                    features = self.backbone(x)
                    return self.projection(features)

                @property
                def conv1(self):
                    # Dummy property for dtype compatibility
                    return self.projection

            # Replace visual encoder
            model.visual = DINOv3Visual(dinov3_backbone, backbone_dim, params['embed_dim'])

            # Freeze backbone if requested (skip if LoRA is handling freezing)
            if freeze_dinov3 and not use_lora:
                for param in model.visual.backbone.parameters():
                    param.requires_grad = False

            print(f"Loaded CLIP model with DINOv3 vision encoder: {dinov3_model_name}"
                  f"{' + LoRA' if use_lora else ''}")
        else:
            print("Loaded in clip model.")
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move entire model to device
    model = model.to(device)
    return model
    
    
def preprocess_text(texts, model):
#     if model.context_length is None: 
#         model = model.module
        
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def setup_validation(config):
    """
    FUNCTION: setup_validation
    ---------------------------------
    This function sets up validation data for training monitoring.
    
    args:
        * config - configuration object with validation parameters
    
    Returns validation loader, ground truth labels, label names, templates, and input resolution
    """
    # Check if validation files exist
    if not os.path.exists(config.val_cxr_filepath) or not os.path.exists(config.val_label_path):
        print("Warning: Validation files not found. Skipping validation setup.")
        return None, None, None, None, None
    
    import zero_shot  # Import here to avoid circular imports
    
    val_labels = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema',
                  'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                  'Lung Opacity', 'No Finding','Pleural Effusion',
                  'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

    # Define standard +/- templates for softmax evaluation
    val_templates = [("{}", "no {}")]  # Using a tuple pair for softmax eval

    try:
        # Load ground truth validation labels
        print(f"Loading validation labels from: {config.val_label_path}")
        y_true_val = zero_shot.make_true_labels(
            cxr_true_labels_path=config.val_label_path,
            cxr_labels=val_labels,
            cutlabels=True  # Keep columns that correspond to val_labels
        )

        # Set input resolution based on model type
        input_resolution = 448 if (not config.random_init or getattr(config, 'use_dinov3', False)) else 320
        print(f"Using validation input resolution: {input_resolution}")

        val_transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])

        # Create validation dataset
        print(f"Loading validation CXR data from: {config.val_cxr_filepath}")
        val_dataset = zero_shot.CXRTestDataset(
            img_path=config.val_cxr_filepath,
            transform=val_transform,
        )

        # Create validation dataloader
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,  # No need to shuffle for validation
            num_workers=2,  # Adjust based on your system
            pin_memory=True
        )

        print("Validation setup complete.")
        return val_loader, y_true_val, val_labels, val_templates, input_resolution
    
    except Exception as e:
        print(f"Warning: Failed to setup validation: {e}")
        return None, None, None, None, None

def make(config, cxr_filepath, txt_filepath, model_path=None): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer. 
    
    args: 
        * config - dict, configuration of experiment
        * cxr_filepath - string, filepath to chest x-ray images
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
    '''
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=config.batch_size, pretrained=config.pretrained, column=config.column)
    model = load_clip(model_path=model_path, pretrained=config.pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    # todo: incorporate - torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    return model, data_loader, device, criterion, optimizer


def train_main(cxr_filepath, txt_filepath, hyperparams, output_path, model_path=None, pretrained=False): 
    '''
    args: 
        * cxr_filpath- str filepath to cxr images
        * txt_filepath- str filepath to text reports
        * hyperparams- dictionary with the following hyperparams:
        `batch_size`, `criterion`, `learning_rate`, `momentum`, `epochs`
        * output_path- str filepath to where the trained model will be saved
        * model_path- str filepath to model that will be used as baseline model for training. 
        If not provided, a model will be trained from scratch
        * pretrained- whether or not the clip model was pretrained with generic images 
    This function is the main train function for CXR-CLIP. 
    '''
    
    # unpack `hyperparams`
    batch_size = hyperparams['batch_size']
    criterion = hyperparams['criterion']
    learning_rate = hyperparams['learning_rate']
    momentum = hyperparams['momentum']
    epochs = hyperparams['epochs']
    
    # load input cxr + report data
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=batch_size, pretrained=pretrained)
    model = load_clip(model_path=model_path, pretrained=pretrained)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_clip(model, data_loader, device, criterion, optimizer, epochs, output_path)
    return model
