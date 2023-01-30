import torch
import torch.nn as nn


# CRNN models
def get_living_mask(x):
    alpha = x[:, 3:4, :, :]
    max_pool = torch.nn.MaxPool2d(kernel_size=25, stride=1, padding=12)
    alpha = max_pool(alpha)
    return alpha > 0.1

def non_liquid_mask(x):
    alpha = x[:, 3:4, :, :]
    return alpha < 0.99


class NCA(torch.nn.Module):
    def __init__(self, parameterization):
        super(NCA, self).__init__()
        self.input_dimension = parameterization.get("in_dim", 16)
        self.neurons = parameterization.get("neu_num", 32)
        self.n_hidden_layers = parameterization.get("hid_lay_num", 3)
        self.output_dimension = parameterization.get("in_dim", 16)
        self.kern_size = parameterization.get("kernel_size", 3)
        self.pad_size = self.kern_size//2 # parameterization.get("pad_size", 1)
        # self.drop_para_1 = parameterization.get("drop_para",0.0)
        self.pad_mode = parameterization.get("padding_mode", 'zeros')
        self.solid_mask_cri = parameterization.get("solid_mask_cri", 0.5)
        self.slow_cri = parameterization.get("slow_cri", 0.5)
        self.speed_red = parameterization.get("speed_red", 1.0)
        self.out_live = parameterization.get("out_live", True)

        self.input_layer = nn.Conv2d(self.input_dimension, self.neurons, kernel_size=self.kern_size, padding=self.pad_size,
                                     padding_mode=self.pad_mode)

        self.hidden_layers = nn.ModuleList(
            [nn.Conv2d(self.neurons, self.neurons, kernel_size=3, padding=1, padding_mode=self.pad_mode)
             for _ in range(self.n_hidden_layers)])

        self.output_layer = nn.Conv2d(self.neurons, self.output_dimension, kernel_size=1, bias=False)
        # self.output_layer.weight.data.zero_()
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        input_x = x

        live_mask = get_living_mask(input_x)
        solid_mask = non_liquid_mask(input_x)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        y1 = self.output_layer(x)
        y = torch.cat([y1[:, :4, ...], 0.0*y1[:, 4:5, ...], y1[:, 5:, ...]], axis=1)
        output_x = input_x + y * live_mask * solid_mask
        return output_x

    def initialize_weights(self):
        torch.nn.init.constant_(self.output_layer.weight.data, 0.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()

def regularization(model,regularization_exp):
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, regularization_exp)
        return reg_loss

def weigt_loss(p, t):
    l = torch.mean(torch.square(p[:, :4, ...] - t[:, :4, ...]), axis=[1, 2, 3])
    l_weight = torch.sum(t[:, 3, ...] == 1.0, axis=[1, 2])
    l = torch.mean(l / (l_weight))
    return l

# load the CRNN Model from file
def load_model(model_file = 'model.pkl'):
    ca = torch.load(model_file, map_location='cpu')
    return ca



