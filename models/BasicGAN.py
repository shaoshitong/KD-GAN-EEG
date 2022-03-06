import torch, math
import torch.nn as nn
import torch.nn.functional as F


# 判别器
class Discriminator(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.disc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_features)
        )
        self.layernorm = nn.LayerNorm(in_features)
        self.classifiar = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.disc(x) + x
        x = self.layernorm(x)
        x = self.classifiar(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features, hidden_dims),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dims, in_features),
        )
        self.classifiar = nn.Linear(in_features, out_features)
        self.layernorm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = self.gen(x) + x
        x = self.layernorm(x)
        x = self.classifiar(x)
        return x


class BasicGAN:
    def __init__(self,
                 in_features, hidden_dims, out_features,
                 generator1=Generator,
                 generator2=Generator,
                 discriminator=Discriminator):
        """
        :param generator1: 源域生成器
        :param generator2: 目标域生成器
        :param discriminator: 辨别器
        :param in_features: 脑电信号输入维度
        :param hidden_dims: 中间层维度
        :param out_features: 辨别器输入维度
        """
        self.source_generator = generator1(in_features, hidden_dims, out_features)
        self.target_generator = generator2(in_features, hidden_dims, out_features)
        self.discriminator = discriminator(out_features, hidden_dims)

    def cuda(self):
        self.source_generator = self.source_generator.cuda()
        self.target_generator = self.target_generator.cuda()
        self.discriminator = self.discriminator.cuda()

    def set_static_parameter(self, model_name: str):
        """
        :param model_name: [source_generator,target_generator,discriminator]
        :return:
        """
        model: nn.Module = getattr(self, model_name)
        model.requires_grad_(False)

    def fit_ont_step_discriminator(self, Source_Sample, Target_Sample):
        """
        """
        if getattr(self, "criterion") == None:
            setattr(self, "criterion", nn.BCELoss())
        Source_Sample = Source_Sample.cuda()
        Target_Sample = Target_Sample.cuda()
        source_number, target_number = Source_Sample.shape[0], Target_Sample.shape[0]
        Source_hidden_vector = self.source_generator(Source_Sample)
        Target_hidden_vector = self.target_generator(Target_Sample)
        Source_value = self.discriminator(Source_hidden_vector).view(-1)
        Target_value = self.discriminator(Target_hidden_vector).view(-1)
        real_loss = -torch.mean(Source_value) * 2
        fake_loss = torch.mean(Target_value) * 2
        return real_loss, fake_loss

    def fit_ont_step_generator(self, Target_Sample):
        if getattr(self, "criterion") == None:
            setattr(self, "criterion", nn.BCELoss())
        Target_Sample = Target_Sample.cuda()
        Target_hidden_vector = self.target_generator(Target_Sample)
        Target_value = self.discriminator(Target_hidden_vector).view(-1)
        fake_loss = -torch.mean(Target_value)
        return fake_loss
