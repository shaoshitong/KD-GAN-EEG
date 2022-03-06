from models.BasicGAN import *
import torch,math
import torch.nn.functional as F
import torch.nn as nn
from timm import scheduler
from torch.utils.data import Dataset,DataLoader

class STGAN(BasicGAN):
    def __init__(self,in_features,hidden_dim,out_features,critic,criterion=nn.BCELoss,
                 generator_lr=1e-5,discriminator_lr=1e-5,epoch=100,warmup_epoch=20):
        super(STGAN,self).__init__(in_features,hidden_dim,out_features)
        self.critic=critic
        self.criterion=nn.BCELoss(size_average=True,reduction="sum")
        self.generator_optimizer=torch.optim.RMSprop(self.target_generator.parameters(),generator_lr)
        self.discriminator_optimizer=torch.optim.RMSprop(self.discriminator.parameters(),discriminator_lr)
        self.set_static_parameter("source_generator")
        self.cuda()
        self.generator_scheduler=scheduler.CosineLRScheduler(self.generator_optimizer,epoch,0,1,warmup_t=warmup_epoch,warmup_lr_init=generator_lr/100)
        self.discriminator_scheduler=scheduler.CosineLRScheduler(self.discriminator_optimizer,epoch,0,1,warmup_t=warmup_epoch,warmup_lr_init=generator_lr/100)
        self.epoch=0
        self.total_epoch=epoch
    def one_epoch_update(self,Source_Sample_Dataloader:DataLoader,Target_Sample_Dataloader:DataLoader):
        self.target_generator.train()
        self.source_generator.train()
        self.discriminator.train()
        metric_disc=0.
        metric_gen=0.
        for source_sample,target_sample in zip(Source_Sample_Dataloader,Target_Sample_Dataloader):
            metric_tmp=0.
            for critic in range(self.critic):
                real_loss,fake_loss=self.fit_ont_step_discriminator(source_sample,target_sample)
                total_loss=(real_loss+fake_loss)/2
                self.discriminator_optimizer.zero_grad()
                total_loss.backward()
                metric_tmp=metric_disc+total_loss.cpu().detach().item()
                self.discriminator_optimizer.step()
                for parameter in self.discriminator.parameters():
                    parameter.data.clamp_(-5., 5.)
            metric_tmp/=self.critic
            metric_disc+=metric_tmp
            fake_loss=self.fit_ont_step_generator(target_sample)
            self.generator_optimizer.zero_grad()
            fake_loss.backward()
            metric_gen=metric_gen+fake_loss.cpu().detach().item()
            self.generator_optimizer.step()
        self.generator_scheduler.step(self.epoch)
        self.discriminator_scheduler.step(self.epoch)
        self.epoch+=1
        dataloader_len=min(len(Source_Sample_Dataloader),len(Target_Sample_Dataloader))
        self.target_generator.eval()
        self.source_generator.eval()
        self.discriminator.eval()
        return metric_disc/dataloader_len,metric_gen/dataloader_len
    def fit(self,Source_Sample_Dataloader:DataLoader,Target_Sample_Dataloader:DataLoader):
        for i in range(self.total_epoch):
            disc_loss,gen_loss=self.one_epoch_update(Source_Sample_Dataloader,Target_Sample_Dataloader)
            print(
                f"Epoch [{i}/{self.total_epoch}] Loss D: {disc_loss:.4f}, loss G: {gen_loss:.4f}"
            )
    def predict(self,Sample):
        with torch.no_grad():
            Sample=Sample.cuda()
            source_mark=self.discriminator(self.source_generator(Sample))
            source_mark=torch.where(source_mark<0,torch.zeros(1).to(Sample.device),source_mark)
            target_mark=-self.discriminator(self.target_generator(Sample))*3
            #target_mark=-torch.where(target_mark>0,torch.zeros(1).to(Sample.device),target_mark)
            target_mark=torch.sigmoid(target_mark.mean())
            # mark=(1e-8+target_mark)/(source_mark+target_mark+2e-8)
            mark=target_mark.detach().cpu()
        return mark





