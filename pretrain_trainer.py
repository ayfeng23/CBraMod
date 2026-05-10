import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn import MSELoss
from torchinfo import summary
from tqdm import tqdm

from utils.util import generate_mask

try:
    import wandb
except ImportError:
    wandb = None


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        if self.params.parallel:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)

        summary(self.model, input_size=(1, 19, 30, 200))

        macs, params = get_model_complexity_info(self.model, (19, 30, 200), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)

        self.use_wandb = bool(getattr(self.params, 'wandb', False)) and wandb is not None
        if self.use_wandb:
            wandb.init(
                project=getattr(self.params, 'wandb_project', 'cbramod-pretrain-tueg'),
                name=getattr(self.params, 'wandb_name', None) or None,
                config=vars(self.params),
            )

        if self.params.lr_scheduler=='CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler=='ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif self.params.lr_scheduler=='StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5*self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler=='MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10*self.data_length, 20*self.data_length, 30*self.data_length], gamma=0.1
            )
        elif self.params.lr_scheduler=='CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001, step_size_up=self.data_length*5,
                step_size_down=self.data_length*2, mode='exp_range', gamma=0.9, cycle_momentum=False
            )


    def train(self):
        best_loss = 10000
        global_step = 0
        recent_grad_norm_max = 0.0
        start_epoch = 0

        resume_path = getattr(self.params, 'resume', None)
        if resume_path:
            print(f"[resume] loading model weights from {resume_path}")
            state = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(state)
            full_path = resume_path + '.full.pth'
            import os
            if os.path.exists(full_path):
                print(f"[resume] loading optimizer/scheduler/epoch from {full_path}")
                full = torch.load(full_path, map_location=self.device)
                self.optimizer.load_state_dict(full['optimizer'])
                self.optimizer_scheduler.load_state_dict(full['scheduler'])
                start_epoch = full['epoch']
                global_step = full['global_step']
                best_loss = full.get('best_loss', best_loss)
            elif getattr(self.params, 'resume_epoch', None) is not None:
                start_epoch = self.params.resume_epoch
                global_step = start_epoch * self.data_length
                for _ in range(global_step):
                    self.optimizer_scheduler.step()
                print(f"[resume] warm-start from epoch {start_epoch} "
                      f"(global_step={global_step}), scheduler fast-forwarded; "
                      f"lr={self.optimizer.param_groups[0]['lr']:.6e}, "
                      f"optimizer state fresh")
            else:
                print("[resume] no .full.pth and no --resume_epoch -- "
                      "warm-start from epoch 0 with fresh optimizer/scheduler")
            if self.params.resume_lr is not None:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.params.resume_lr
                print(f"[resume] LR overridden to {self.params.resume_lr}")

        for epoch in range(start_epoch, self.params.epochs):
            losses = []
            for x in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)/100
                if self.params.need_mask:
                    bz, ch_num, patch_num, patch_size = x.shape
                    mask = generate_mask(
                        bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device,
                    )
                    y = self.model(x, mask=mask)
                    masked_x = x[mask == 1]
                    masked_y = y[mask == 1]
                    loss = self.criterion(masked_y, masked_x)
                elif self.params.objective == "recon":
                    y = self.model(x)
                    loss = self.criterion(y, x)
                elif self.params.objective == "ntp":
                    y = self.model(x[:, :, :-1, :])
                    labels = x[:, :, 1:, :]
                    loss = self.criterion(y, labels)
                loss.backward()
                # clip_grad_norm_ returns the *pre-clip* total norm; capture it for logging.
                # Pass inf when clipping is disabled so we still get the norm without scaling.
                max_norm = self.params.clip_value if self.params.clip_value > 0 else float('inf')
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm))
                if grad_norm > recent_grad_norm_max:
                    recent_grad_norm_max = grad_norm
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())
                global_step += 1
                if self.use_wandb and global_step % 50 == 0:
                    wandb.log({
                        'train/loss': float(loss.item()),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/grad_norm': grad_norm,
                        'train/grad_norm_max_recent': recent_grad_norm_max,
                        'epoch': epoch + (global_step % self.data_length) / max(self.data_length, 1),
                    }, step=global_step)
                    recent_grad_norm_max = 0.0
            mean_loss = np.mean(losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}')
            if self.use_wandb:
                wandb.log({
                    'epoch/mean_loss': float(mean_loss),
                    'epoch/lr': learning_rate,
                    'epoch/index': epoch + 1,
                }, step=global_step)
            if  mean_loss < best_loss or (epoch % 5 == 0):
                model_path = rf'{self.params.model_dir}/epoch{epoch+1}_loss{mean_loss}.pth'
                torch.save(self.model.state_dict(), model_path)
                torch.save({
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.optimizer_scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'best_loss': float(best_loss),
                }, model_path + '.full.pth')
                print("model save in " + model_path)
                best_loss = mean_loss
                if mean_loss < best_loss:
                    if self.use_wandb:
                        wandb.summary['best_loss'] = float(best_loss)
                        wandb.summary['best_epoch'] = epoch + 1
        if self.use_wandb:
            wandb.finish()