import os
import argparse
import h5py
import pandas as pd
import numpy as np
from math import pi
from tqdm import tqdm

import clip
import torch
import torch.optim as optim
from torch import nn
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from eval import evaluate
from train import load_data, load_clip, preprocess_text, setup_validation



class MultiCXRDataset(data.Dataset):
    def __init__(self, dataset_paths, column='impression', transform=None):
        super().__init__()
        self.datasets = []
        self.dataset_lengths = []
        self.cumulative_lengths = []
        self.transform = transform
        
        cumulative_length = 0
        for path_pair in dataset_paths:
            img_path, txt_path = path_pair.split(',')
            img_dset = h5py.File(img_path, 'r')['cxr']
            txt_dset = pd.read_csv(txt_path)[column]
            
            # Ensure image-text pairing is correct
            assert len(img_dset) == len(txt_dset), f"Mismatch in {img_path} and {txt_path}: {len(img_dset)} vs {len(txt_dset)}"
            
            self.datasets.append((img_dset, txt_dset))
            dataset_len = len(txt_dset)
            self.dataset_lengths.append(dataset_len)
            cumulative_length += dataset_len
            self.cumulative_lengths.append(cumulative_length)
            
        print(f"Loaded {len(self.datasets)} datasets with total {cumulative_length} samples")
        for i, (length, path_pair) in enumerate(zip(self.dataset_lengths, dataset_paths)):
            print(f"  Dataset {i+1}: {length} samples from {path_pair}")
    
    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Find which dataset this index belongs to
        dataset_idx = 0
        local_idx = idx
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                if i > 0:
                    local_idx = idx - self.cumulative_lengths[i-1]
                break
        
        img_dset, txt_dset = self.datasets[dataset_idx]
        img = img_dset[local_idx]
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        txt = txt_dset.iloc[local_idx]
        
        if pd.isna(txt):
            txt = " "
        
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        
        return {'img': img, 'txt': txt}

def load_multi_data(dataset_paths, batch_size=64, column='impression', pretrained=False, use_dinov3=False, use_ddp=False, rank=0):
    """
    Load multiple CXR datasets with DDP support.
    
    Args:
        dataset_paths: List of 'img_path,txt_path' pairs
        batch_size: Batch size per GPU
        column: Text column to use from CSV
        pretrained: Whether using pretrained models
        use_dinov3: Whether using DINOv3 vision encoder
        use_ddp: Whether using distributed training
        rank: GPU rank for DDP
    
    Returns:
        data_loader: DataLoader with DistributedSampler if DDP enabled
        device: CUDA device for this rank
    """
    if torch.cuda.is_available():
        if use_ddp:
            dev = f"cuda:{rank}"
            device = torch.device(dev)
            torch.cuda.set_device(device)
        else:
            dev = "cuda:0"
            device = torch.device(dev)
            torch.cuda.set_device(device)
        cuda_available = True
        print(f'Using CUDA device: {dev}')
    else:
        dev = "cpu"
        cuda_available = False
        device = torch.device(dev)
        print('Using cpu.')
    
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
    
    torch_dset = MultiCXRDataset(dataset_paths=dataset_paths, column=column, transform=transform)
    
    # Configure loader parameters for DDP
    if use_ddp:
        sampler = DistributedSampler(torch_dset, num_replicas=None, rank=None, shuffle=True)
        loader_params = {'batch_size': batch_size, 'sampler': sampler, 'num_workers': 8, 'pin_memory': True}
    else:
        loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
    
    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='/cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train.h5')
    parser.add_argument('--txt_filepath', type=str, default='/cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train_metadata.csv')
    parser.add_argument('--use_multi_datasets', action='store_true', help='Use multiple CXR-report datasets for training')
    parser.add_argument('--dataset_paths', type=str, nargs='+',
                        default=['/cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train.h5,/cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_train_metadata.csv',
                                '/cbica/projects/CXR/data_p/rexgradient/rexgradient_train.h5,/cbica/projects/CXR/data_p/rexgradient/rexgradient_train_metadata.csv'],
                        help='List of dataset paths in format: img_path,txt_path')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--warmup_steps', type=int, default=250)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true', default=True)
    parser.add_argument('--no_random_init', dest='random_init', action='store_false',
                        help='Use pretrained CLIP weights instead of random initialization')
    parser.add_argument('--model_name', type=str, default="dinov3-multi-v1.0")
    parser.add_argument('--do_validate', action='store_true', help='Enable validation during training')
    parser.add_argument('--valid_interval', type=int, default=200)
    parser.add_argument('--val_cxr_filepath', type=str, default='/cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_valid.h5')
    parser.add_argument('--val_label_path', type=str, default='data/chexpert_valid.csv')
    parser.add_argument('--val_batch_size', type=int, default=64)
    # Test dataset arguments - added for final evaluation
    parser.add_argument('--test_after_training', action='store_true', help='Test on CheXpert and PadChest after training')
    parser.add_argument('--chexpert_test_cxr', type=str, default='/cbica/projects/CXR/data_p/chexpert-plus/chexpert_plus_test.h5', help='CheXpert test images')
    parser.add_argument('--chexpert_test_labels', type=str, default='data/chexpert_test.csv', help='CheXpert test labels')
    parser.add_argument('--padchest_test_cxr', type=str, default='', help='PadChest test images (optional)')
    parser.add_argument('--padchest_test_labels', type=str, default='', help='PadChest test labels (optional)')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size for testing')
    # DINOv3 specific arguments
    parser.add_argument('--use_dinov3', action='store_true', help='Use DINOv3 as vision encoder')
    parser.add_argument('--dinov3_model_name', type=str, default='dinov3_vitb16',
                        choices=['dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16',
                                 'dinov3_vitl16', 'dinov3_vith16plus', 'dinov3_vit7b16'],
                        help='DINOv3 model variant to use')
    parser.add_argument('--dinov3_repo_dir', type=str,
                        default='/cbica/projects/CXR/codes/dinov3',
                        help='Path to local DINOv3 repo')
    parser.add_argument('--dinov3_weights', type=str,
                        default='/cbica/projects/CXR/codes/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                        help='Path to DINOv3 pretrained weights')
    parser.add_argument('--freeze_dinov3', action='store_true', help='Freeze DINOv3 backbone weights')
    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait without improvement before stopping')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as an improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='mean_auc', 
                        choices=['mean_auc', 'loss'], help='Metric to use for early stopping')
    # DDP arguments
    parser.add_argument('--use_ddp', action='store_true', help='Use Distributed Data Parallel training')
    parser.add_argument('--backend', type=str, default='nccl', help='DDP backend')
    args = parser.parse_args()
    return args

def setup_ddp(backend='nccl'):
    """Initialize the distributed environment using torchrun."""
    try:
        dist.init_process_group(backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        print(f"Successfully initialized DDP on rank {rank}/{world_size}")
        return rank, world_size
    except Exception as e:
        print(f"Failed to initialize DDP: {e}")
        raise

def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def get_vit_variant(config):
    """
    Helper function to extract the ViT variant from DINOv3 model name.
    Returns the ViT variant (e.g., 'vitb', 'vits', 'vitl') or None for other models.
    """
    if config.use_dinov3:
        # Extract ViT variant from dinov3 model name
        # e.g., dinov3_vitb16 -> vitb, dinov3_vits16plus -> vits16plus
        model_parts = config.dinov3_model_name.split('_')
        if len(model_parts) >= 2:
            vit_part = model_parts[1]  # e.g., 'vitb16' or 'vits16plus'
            # Remove 'dinov3_' prefix already handled by split, return rest
            return vit_part
    return None

def model_pipeline(config, verbose=0):
    if config.use_ddp:
        # DDP training (launched with torchrun)
        ddp_main(config, verbose)
    else:
        # Regular single GPU training
        single_gpu_pipeline(config, verbose)

def ddp_main(config, verbose=0):
    """Main function for DDP training (called by torchrun)."""
    
    # Initialize DDP and get rank/world_size
    rank, world_size = setup_ddp(config.backend)
    print(f"Running DDP process on rank {rank}")
    
    # Set random seed for reproducibility (same seed for all ranks)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Modify model_name to include ViT variant for DINOv3 models
    vit_variant = get_vit_variant(config)
    if vit_variant:
        original_model_name = config.model_name
        config.model_name = f"{original_model_name}_{vit_variant}"
        if rank == 0:  # Only print from main process
            print(f"Using checkpoint folder: {config.model_name} (ViT variant: {vit_variant})")
    else:
        if rank == 0:
            print(f"Using checkpoint folder: {config.model_name}")

    try:
        model, data_loader, device, criterion, optimizer, scheduler, scaler = make(config, rank)
        train(model, data_loader, device, criterion, optimizer, scheduler, scaler, config, rank)

        # Save model only from rank 0
        if rank == 0:
            model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
            # Extract the actual model from DDP wrapper for saving
            model_to_save = model.module if hasattr(model, 'module') else model
            save(model_to_save, model_path)

            # Run final testing if requested
            if config.test_after_training:
                run_final_testing(config)

        if verbose and rank == 0:
            print(model)
            
    finally:
        cleanup_ddp()

def single_gpu_pipeline(config, verbose=0):
    """Original single GPU pipeline."""
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Modify model_name to include ViT variant for DINOv3 models
    vit_variant = get_vit_variant(config)
    if vit_variant:
        original_model_name = config.model_name
        config.model_name = f"{original_model_name}_{vit_variant}"
        print(f"Using checkpoint folder: {config.model_name} (ViT variant: {vit_variant})")
    else:
        print(f"Using checkpoint folder: {config.model_name}")

    model, data_loader, device, criterion, optimizer, scheduler, scaler = make(config)
    train(model, data_loader, device, criterion, optimizer, scheduler, scaler, config)

    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    # Run final testing if requested
    if config.test_after_training:
        run_final_testing(config)

    if verbose:
        print(model)
    return model

def make(config, rank=0):
    pretrained = not config.random_init
    
    if config.use_multi_datasets:
        data_loader, device = load_multi_data(
            dataset_paths=config.dataset_paths,
            batch_size=config.batch_size,
            column="impression",
            pretrained=pretrained,
            use_dinov3=config.use_dinov3,
            use_ddp=config.use_ddp,
            rank=rank
        )
    else:
        # For single dataset, use multi_data with single path when DDP is enabled
        if config.use_ddp:
            dataset_paths = [f"{config.cxr_filepath},{config.txt_filepath}"]
            data_loader, device = load_multi_data(
                dataset_paths=dataset_paths,
                batch_size=config.batch_size,
                column="impression",
                pretrained=pretrained,
                use_dinov3=config.use_dinov3,
                use_ddp=config.use_ddp,
                rank=rank
            )
        else:
            data_loader, device = load_data(
                config.cxr_filepath, config.txt_filepath, 
                batch_size=config.batch_size, 
                pretrained=pretrained, 
                use_dinov3=config.use_dinov3,
                column="impression"
            )
    
    model = load_clip(
        model_path=None,
        pretrained=pretrained,
        context_length=config.context_length,
        use_dinov3=config.use_dinov3,
        dinov3_model_name=config.dinov3_model_name,
        dinov3_repo_dir=config.dinov3_repo_dir,
        dinov3_weights=config.dinov3_weights,
        freeze_dinov3=config.freeze_dinov3
    )
    model.to(device)
    
    # Wrap model with DDP if enabled
    if config.use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        print(f'Model wrapped with DDP on rank {rank}.')
    else:
        print('Model on Device.')

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))

    total_steps = config.epochs * len(data_loader)
    if config.lr_schedule == 'cosine':
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            progress = float(current_step - config.warmup_steps) / float(max(1, total_steps - config.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * pi))))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda")
    return model, data_loader, device, criterion, optimizer, scheduler, scaler

def train(model, loader, device, criterion, optimizer, scheduler, scaler, config, rank=0):
    model.train()
    
    # Only setup validation on rank 0 to avoid conflicts
    if rank == 0:
        val_loader, y_true_val, val_labels, val_templates, _ = setup_validation(config)
        validation_enabled = val_loader is not None
        
        model_save_dir = os.path.join(config.save_dir, config.model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        # Initialize validation log file
        val_log_path = os.path.join(model_save_dir, "validation_log.txt")
        with open(val_log_path, 'w') as f:
            f.write("Step,Epoch,Mean_AUC,Atelectasis_AUC,Cardiomegaly_AUC,Consolidation_AUC,Edema_AUC,Pleural_Effusion_AUC\n")

        # Best model tracking variables 
        best_metric = float('-inf') if config.early_stopping_metric == 'mean_auc' else float('inf')
        epochs_without_improvement = 0
        best_epoch = 0
        best_step = 0
    else:
        validation_enabled = False

    example_ct = 0
    batch_ct = 0
    running_loss = 0.0

    optimizer.zero_grad()

    for epoch in range(config.epochs):
        # Set epoch for distributed sampler to ensure proper data shuffling
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
            
        for data in tqdm(loader, disable=(rank != 0)):  # Only show progress bar on rank 0
            images = data['img'].to(device)
            # Get the actual model for text preprocessing (unwrap DDP if needed)
            model_for_text = model.module if hasattr(model, 'module') else model
            texts = preprocess_text(data['txt'], model_for_text).to(device)

            with torch.amp.autocast('cuda'):
                logits_per_image, logits_per_text = model(images, texts)
                labels = torch.arange(images.size(0), device=device)
                loss_img = criterion(logits_per_image, labels)
                loss_txt = criterion(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2

            scaler.scale(loss).backward()

            if (batch_ct + 1) % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler: scheduler.step()

            example_ct += images.size(0)
            batch_ct += 1
            running_loss += loss.item()

            # Only log and validate from rank 0
            if rank == 0:
                if batch_ct % config.log_interval == 0:
                    train_log(running_loss / config.log_interval, example_ct, epoch)
                    running_loss = 0.0

                if config.do_validate and validation_enabled and (batch_ct % config.valid_interval) == 0:
                    # Get the actual model for validation (unwrap DDP if needed)
                    model_for_validation = model.module if hasattr(model, 'module') else model
                    val_results_df = run_validation_step(model_for_validation, val_loader, y_true_val, val_labels, val_templates, device, config)
                    
                    # Calculate mean AUC for key pathologies
                    key_pathologies = ['Atelectasis_auc', 'Cardiomegaly_auc', 'Consolidation_auc', 'Edema_auc', 'Pleural Effusion_auc']
                    available_cols = [col for col in key_pathologies if col in val_results_df.columns]
                    if available_cols:
                        current_auc = val_results_df[available_cols].mean().mean()
                    else:
                        auc_cols = [col for col in val_results_df.columns if col.endswith('_auc')]
                        current_auc = val_results_df[auc_cols].mean().mean() if auc_cols else 0
                    
                    # Log validation results
                    with open(val_log_path, 'a') as f:
                        auc_values = [val_results_df[col].iloc[0] if col in val_results_df.columns else 0 for col in key_pathologies]
                        f.write(f"{batch_ct},{epoch},{current_auc:.4f},{','.join(f'{v:.4f}' for v in auc_values)}\n")
                    
                    print(f"Validation at step {batch_ct}: Mean AUC = {current_auc:.4f}")
                    
                    # Check if this is the best model so far
                    if current_auc > best_metric + config.min_delta:
                        best_metric = current_auc
                        best_step = batch_ct
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        # Save best model (unwrap DDP if needed)
                        best_model_path = os.path.join(model_save_dir, "best_model.pt")
                        model_to_save = model.module if hasattr(model, 'module') else model
                        save(model_to_save, best_model_path)
                        print(f"New best model saved! AUC: {current_auc:.4f} at step {batch_ct}")
                    else:
                        epochs_without_improvement += 1
        
        # Early stopping check at the end of each epoch (DDP-safe)
        should_stop = False
        if config.early_stopping and rank == 0:
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping triggered! No improvement for {config.patience} validation intervals.")
                print(f"Best mean AUC: {best_metric:.4f} achieved at step {best_step} (epoch {best_epoch})")
                should_stop = True
        
        # Synchronize early stopping decision across all processes
        if config.use_ddp:
            # Broadcast early stopping decision from rank 0 to all ranks
            stop_tensor = torch.tensor(should_stop, dtype=torch.bool, device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item()
            dist.barrier()
        
        if should_stop:
            break
        
        # Reset running loss for next epoch
        running_loss = 0.0

def train_log(loss, example_ct, epoch):
    print(f"Loss after {str(example_ct).zfill(5)} examples (Epoch {epoch}): {loss:.3f}")

def run_validation_step(model, val_loader, y_true_val, val_labels, val_templates, device, config):
    model.eval()
    context_length = getattr(model, 'context_length', config.context_length)
    pos_template, neg_template = val_templates[0]

    with torch.no_grad():
        pos_texts = [pos_template.format(c) for c in val_labels]
        neg_texts = [neg_template.format(c) for c in val_labels]
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)

    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation Inference"):
            imgs = data['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())

    img_feats_cat = torch.cat(all_img_feats).to(device)
    logits_pos = img_feats_cat @ pos_features.T
    logits_neg = img_feats_cat @ neg_features.T
    probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    y_pred_val = probs.cpu().numpy()
    val_results_df = evaluate(y_pred_val, y_true_val, val_labels)
    model.train()
    return val_results_df

def save(model, path):
    torch.save(model.state_dict(), path)

# ====================== TESTING FUNCTIONS - Added for final evaluation ======================

def find_best_model(config):
    """Find the best model saved during training."""
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    
    # Check if best model exists (should always exist with new logging system)
    best_model_path = os.path.join(model_save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        # Read validation log to get the best AUC score
        val_log_path = os.path.join(model_save_dir, "validation_log.txt")
        if os.path.exists(val_log_path):
            try:
                df = pd.read_csv(val_log_path)
                best_idx = df['Mean_AUC'].idxmax()
                best_auc = df.loc[best_idx, 'Mean_AUC']
                best_step = df.loc[best_idx, 'Step']
                print(f"Using best model: AUC = {best_auc:.4f} at step {best_step}")
            except:
                print("Using best model (unable to read validation log)")
        else:
            print("Using best model from training")
        return best_model_path
    
    # Fallback to final checkpoint if best model doesn't exist
    final_checkpoint = os.path.join(model_save_dir, 'checkpoint.pt')
    if os.path.exists(final_checkpoint):
        print("Warning: Best model not found. Using final checkpoint.")
        return final_checkpoint
    
    raise FileNotFoundError(f"No model found in {model_save_dir}")

def setup_test_dataset(test_cxr_filepath, test_label_path, labels, config):
    """Setup test dataset loader and ground truth labels."""
    import zero_shot
    from torchvision.transforms import InterpolationMode
    
    print(f"Loading test labels from: {test_label_path}")
    y_true_test = zero_shot.make_true_labels(
        cxr_true_labels_path=test_label_path,
        cxr_labels=labels,
        cutlabels=True
    )
    
    # Use same resolution as validation
    input_resolution = 448 if (not config.random_init or getattr(config, 'use_dinov3', False)) else 320
    
    test_transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
    ])
    
    print(f"Loading test CXR data from: {test_cxr_filepath}")
    test_dataset = zero_shot.CXRTestDataset(
        img_path=test_cxr_filepath,
        transform=test_transform,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader, y_true_test

def test_model_on_dataset(model, test_loader, y_true_test, labels, templates, device, config, dataset_name):
    """Test model on a specific dataset and return results."""
    model.eval()
    context_length = getattr(model, 'context_length', config.context_length)
    pos_template, neg_template = templates[0]
    
    print(f"\n=== Testing on {dataset_name} ===")
    
    # Encode text templates
    with torch.no_grad():
        pos_texts = [pos_template.format(c) for c in labels]
        neg_texts = [neg_template.format(c) for c in labels]
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    
    # Extract image features
    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            imgs = data['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())
    
    # Compute predictions and evaluate
    img_feats_cat = torch.cat(all_img_feats).to(device)
    logits_pos = img_feats_cat @ pos_features.T
    logits_neg = img_feats_cat @ neg_features.T
    probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    y_pred_test = probs.cpu().numpy()
    
    test_results_df = evaluate(y_pred_test, y_true_test, labels)
    return test_results_df

def run_final_testing(config):
    """Run testing on both CheXpert and PadChest test datasets using the best model."""
    print("\n" + "="*60)
    print("STARTING FINAL TESTING ON TEST DATASETS")
    print("="*60)
    
    # Find best model
    best_model_path = find_best_model(config)
    
    # Load the best model
    model = load_clip(
        model_path=best_model_path,
        pretrained=not config.random_init,
        context_length=config.context_length,
        use_dinov3=config.use_dinov3,
        dinov3_model_name=config.dinov3_model_name,
        dinov3_repo_dir=config.dinov3_repo_dir,
        dinov3_weights=config.dinov3_weights,
        freeze_dinov3=config.freeze_dinov3
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    results_dir = os.path.join(config.save_dir, config.model_name, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Test on CheXpert
    if os.path.exists(config.chexpert_test_cxr) and os.path.exists(config.chexpert_test_labels):
        chexpert_labels = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema',
                          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                          'Lung Opacity', 'No Finding','Pleural Effusion',
                          'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        chexpert_templates = [("{}", "no {}")]
        
        chexpert_loader, y_true_chexpert = setup_test_dataset(
            config.chexpert_test_cxr, config.chexpert_test_labels, chexpert_labels, config)
        
        chexpert_results = test_model_on_dataset(
            model, chexpert_loader, y_true_chexpert, chexpert_labels, 
            chexpert_templates, device, config, "CheXpert Test")
        
        chexpert_results.to_csv(os.path.join(results_dir, "chexpert_test_results.csv"), index=False)
        print(f"CheXpert test results saved to: {results_dir}/chexpert_test_results.csv")
        
        # Print key results
        key_cols = ['Atelectasis_auc', 'Cardiomegaly_auc', 'Consolidation_auc', 'Edema_auc', 'Pleural Effusion_auc']
        available_cols = [col for col in key_cols if col in chexpert_results.columns]
        if available_cols:
            mean_auc = chexpert_results[available_cols].mean().mean()
            print(f"CheXpert Mean AUC (key pathologies): {mean_auc:.4f}")
    
    # Test on PadChest  
    if os.path.exists(config.padchest_test_cxr) and os.path.exists(config.padchest_test_labels):
        # Read PadChest labels from CSV
        df_padchest = pd.read_csv(config.padchest_test_labels)
        if 'is_test' in df_padchest.columns:
            df_padchest = df_padchest[df_padchest['is_test'] == True]
        
        # Get disease labels (excluding ImageID, name, Path, is_test columns)
        exclude_cols = ['ImageID', 'name', 'Path', 'is_test']
        padchest_labels = [col.lower() for col in df_padchest.columns if col not in exclude_cols]
        padchest_templates = [("{}", "no {}")]
        
        padchest_loader, y_true_padchest = setup_test_dataset(
            config.padchest_test_cxr, config.padchest_test_labels, padchest_labels, config)
        
        padchest_results = test_model_on_dataset(
            model, padchest_loader, y_true_padchest, padchest_labels,
            padchest_templates, device, config, "PadChest Test")
        
        padchest_results.to_csv(os.path.join(results_dir, "padchest_test_results.csv"), index=False)
        print(f"PadChest test results saved to: {results_dir}/padchest_test_results.csv")
        
        # Print summary
        mean_auc = padchest_results.mean(axis=1).iloc[0] if len(padchest_results) > 0 else 0
        print(f"PadChest Mean AUC (all pathologies): {mean_auc:.4f}")
    
    print("\n" + "="*60)
    print("FINAL TESTING COMPLETED")
    print("="*60)

# ====================== END TESTING FUNCTIONS ======================

if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
