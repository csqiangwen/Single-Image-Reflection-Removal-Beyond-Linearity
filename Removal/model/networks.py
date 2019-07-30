import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'reflrmnetwork':
        netG = ReflRmNetwork(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


class DownSamplingBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 innermost=False, outmost=False):
        super(DownSamplingBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(output_nc)
        model = []

        if innermost:
            model += [downrelu, downconv]
        elif outmost:
            model += [downconv]
        else:
            model += [downrelu, downconv, downnorm]
        if use_dropout:
                model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)


class UpSamplingBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 innermost=False, outmost=False, w_state=False):
        super(UpSamplingBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        upconv = nn.ConvTranspose2d(input_nc, output_nc,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(output_nc)
        model = []

        if innermost:
            model += [uprelu, upconv, upnorm]
        elif outmost and w_state:
            upconv = nn.ConvTranspose2d(input_nc, output_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)
            model += [uprelu, upconv, nn.Sigmoid()]
        elif outmost:
            upconv = nn.ConvTranspose2d(input_nc, output_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)
            model += [uprelu, upconv, nn.Tanh()]
        else:
            model += [uprelu, upconv, upnorm]
        if use_dropout:
                model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ReflRmNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(ReflRmNetwork, self).__init__()
        self.gpu_ids=gpu_ids

        # Downsampling
        block_d_0 = [DownSamplingBlock(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, outmost=True)]
        block_d_1 = [DownSamplingBlock(ngf, ngf*2, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_d_2 = [DownSamplingBlock(ngf*2, ngf*4, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_d_3 = [DownSamplingBlock(ngf*4, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_d_4 = [DownSamplingBlock(ngf*8, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_d_inner = [DownSamplingBlock(ngf*8, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True)]

        # Upsampling for transmission ('back' is for the old name 'background')
        block_u_inner_back = [UpSamplingBlock(ngf*8, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True)]
        block_u_4_back = [UpSamplingBlock(ngf*16, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_3_back = [UpSamplingBlock(ngf*16, ngf*4, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_2_back = [UpSamplingBlock(ngf*8, ngf*2, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_1_back = [UpSamplingBlock(ngf*4, ngf, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_0_back = [UpSamplingBlock(ngf*2, output_nc, norm_layer=norm_layer, use_dropout=use_dropout, outmost=True)]

        #Upsampling for reflection
        block_u_inner_refl = [UpSamplingBlock(ngf*8, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True)]
        block_u_4_refl = [UpSamplingBlock(ngf*16, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_3_refl = [UpSamplingBlock(ngf*16, ngf*4, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_2_refl = [UpSamplingBlock(ngf*8, ngf*2, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_1_refl = [UpSamplingBlock(ngf*4, ngf, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_0_refl = [UpSamplingBlock(ngf*2, output_nc, norm_layer=norm_layer, use_dropout=use_dropout, outmost=True)]

        # Upsampling for W
        block_u_inner_W = [UpSamplingBlock(ngf*8, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True)]
        block_u_4_W = [UpSamplingBlock(ngf*16, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_3_W = [UpSamplingBlock(ngf*16, ngf*4, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_2_W = [UpSamplingBlock(ngf*8, ngf*2, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_1_W = [UpSamplingBlock(ngf*4, ngf, norm_layer=norm_layer, use_dropout=use_dropout)]
        block_u_0_W = [UpSamplingBlock(ngf*2, output_nc, norm_layer=norm_layer, use_dropout=use_dropout, outmost=True ,w_state=True)]

        # model for downsampling
        self.model_d_0 = nn.Sequential(*block_d_0)
        self.model_d_1 = nn.Sequential(*block_d_1)
        self.model_d_2 = nn.Sequential(*block_d_2)
        self.model_d_3 = nn.Sequential(*block_d_3)
        self.model_d_4 = nn.Sequential(*block_d_4)
        self.model_d_inner = nn.Sequential(*block_d_inner)

        # model for transmission upsampling ('back' is for the old name 'background')
        self.model_u_0_back = nn.Sequential(*block_u_0_back)
        self.model_u_1_back = nn.Sequential(*block_u_1_back)
        self.model_u_2_back = nn.Sequential(*block_u_2_back)
        self.model_u_3_back = nn.Sequential(*block_u_3_back)
        self.model_u_4_back = nn.Sequential(*block_u_4_back)
        self.model_u_inner_back = nn.Sequential(*block_u_inner_back)

        # model for reflection upsampling
        self.model_u_0_refl = nn.Sequential(*block_u_0_refl)
        self.model_u_1_refl = nn.Sequential(*block_u_1_refl)
        self.model_u_2_refl = nn.Sequential(*block_u_2_refl)
        self.model_u_3_refl = nn.Sequential(*block_u_3_refl)
        self.model_u_4_refl = nn.Sequential(*block_u_4_refl)
        self.model_u_inner_refl = nn.Sequential(*block_u_inner_refl)

        # model for W upsampling
        self.model_u_0_W = nn.Sequential(*block_u_0_W)
        self.model_u_1_W = nn.Sequential(*block_u_1_W)
        self.model_u_2_W = nn.Sequential(*block_u_2_W)
        self.model_u_3_W = nn.Sequential(*block_u_3_W)
        self.model_u_4_W = nn.Sequential(*block_u_4_W)
        self.model_u_inner_W = nn.Sequential(*block_u_inner_W)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            # downsampling features
            feature_d_0 = nn.parallel.data_parallel(self.model_d_0, input, self.gpu_ids)
            feature_d_1 = nn.parallel.data_parallel(self.model_d_1, feature_d_0, self.gpu_ids)
            feature_d_2 = nn.parallel.data_parallel(self.model_d_2, feature_d_1, self.gpu_ids)
            feature_d_3 = nn.parallel.data_parallel(self.model_d_3, feature_d_2, self.gpu_ids)
            feature_d_4 = nn.parallel.data_parallel(self.model_d_4, feature_d_3, self.gpu_ids)
            feature_d_inner = nn.parallel.data_parallel(self.model_d_inner, feature_d_4, self.gpu_ids)

            # upsampling transmission features ('back' is for the old name 'background')
            feature_u_4_back =  nn.parallel.data_parallel(self.model_u_inner_back, feature_d_inner, self.gpu_ids)
            horizontal_4 = feature_d_4.shape[2] - feature_u_4_back.shape[2]
            vertical_4 = feature_d_4.shape[3] - feature_u_4_back.shape[3]
            feature_u_4_back = torch.nn.functional.pad(feature_u_4_back, (vertical_4, 0, horizontal_4, 0))

            feature_u_3_back =  nn.parallel.data_parallel(self.model_u_4_back, torch.cat([feature_d_4, feature_u_4_back], 1), self.gpu_ids)
            horizontal_3 = feature_d_3.shape[2] - feature_u_3_back.shape[2]
            vertical_3 = feature_d_3.shape[3] - feature_u_3_back.shape[3]
            feature_u_3_back = torch.nn.functional.pad(feature_u_3_back, (vertical_3, 0, horizontal_3, 0))

            feature_u_2_back =  nn.parallel.data_parallel(self.model_u_3_back, torch.cat([feature_d_3, feature_u_3_back], 1), self.gpu_ids)
            horizontal_2 = feature_d_2.shape[2] - feature_u_2_back.shape[2]
            vertical_2 = feature_d_2.shape[3] - feature_u_2_back.shape[3]
            feature_u_2_back = torch.nn.functional.pad(feature_u_2_back, (vertical_2, 0, horizontal_2, 0))

            feature_u_1_back =  nn.parallel.data_parallel(self.model_u_2_back, torch.cat([feature_d_2, feature_u_2_back], 1), self.gpu_ids)
            horizontal_1 = feature_d_1.shape[2] - feature_u_1_back.shape[2]
            vertical_1 = feature_d_1.shape[3] - feature_u_1_back.shape[3]
            feature_u_1_back = torch.nn.functional.pad(feature_u_1_back, (vertical_1, 0, horizontal_1, 0))

            feature_u_0_back =  nn.parallel.data_parallel(self.model_u_1_back, torch.cat([feature_d_1, feature_u_1_back], 1), self.gpu_ids)
            horizontal_0 = feature_d_0.shape[2] - feature_u_0_back.shape[2]
            vertical_0 = feature_d_0.shape[3] - feature_u_0_back.shape[3]
            feature_u_0_back = torch.nn.functional.pad(feature_u_0_back, (vertical_0, 0, horizontal_0, 0))

            output_trans =  nn.parallel.data_parallel(self.model_u_0_back, torch.cat([feature_d_0, feature_u_0_back], 1), self.gpu_ids)

            # unsampling reflection features
            feature_u_4_refl =  nn.parallel.data_parallel(self.model_u_inner_refl, feature_d_inner, self.gpu_ids)
            feature_u_4_refl = torch.nn.functional.pad(feature_u_4_refl, (vertical_4, 0, horizontal_4, 0))

            feature_u_3_refl =  nn.parallel.data_parallel(self.model_u_4_refl, torch.cat([feature_d_4, feature_u_4_refl], 1), self.gpu_ids)
            feature_u_3_refl = torch.nn.functional.pad(feature_u_3_refl, (vertical_3, 0, horizontal_3, 0))

            feature_u_2_refl =  nn.parallel.data_parallel(self.model_u_3_refl, torch.cat([feature_d_3, feature_u_3_refl], 1), self.gpu_ids)
            feature_u_2_refl = torch.nn.functional.pad(feature_u_2_refl, (vertical_2, 0, horizontal_2, 0))

            feature_u_1_refl =  nn.parallel.data_parallel(self.model_u_2_refl, torch.cat([feature_d_2, feature_u_2_refl], 1), self.gpu_ids)
            feature_u_1_refl = torch.nn.functional.pad(feature_u_1_refl, (vertical_1, 0, horizontal_1, 0))

            feature_u_0_refl =  nn.parallel.data_parallel(self.model_u_1_refl, torch.cat([feature_d_1, feature_u_1_refl], 1), self.gpu_ids)
            feature_u_0_refl = torch.nn.functional.pad(feature_u_0_refl, (vertical_0, 0, horizontal_0, 0))

            output_refl =  nn.parallel.data_parallel(self.model_u_0_refl, torch.cat([feature_d_0, feature_u_0_refl], 1), self.gpu_ids)

            # upsampling W features
            feature_u_4_W =  nn.parallel.data_parallel(self.model_u_inner_W, feature_d_inner, self.gpu_ids)
            feature_u_4_W = torch.nn.functional.pad(feature_u_4_W, (vertical_4, 0, horizontal_4, 0))

            feature_u_3_W =  nn.parallel.data_parallel(self.model_u_4_W, torch.cat([feature_d_4, feature_u_4_W], 1), self.gpu_ids)
            feature_u_3_W = torch.nn.functional.pad(feature_u_3_W, (vertical_3, 0, horizontal_3, 0))

            feature_u_2_W =  nn.parallel.data_parallel(self.model_u_3_W, torch.cat([feature_d_3, feature_u_3_W], 1), self.gpu_ids)
            feature_u_2_W = torch.nn.functional.pad(feature_u_2_W, (vertical_2, 0, horizontal_2, 0))

            feature_u_1_W =  nn.parallel.data_parallel(self.model_u_2_W, torch.cat([feature_d_2, feature_u_2_W], 1), self.gpu_ids)
            feature_u_1_W = torch.nn.functional.pad(feature_u_1_W, (vertical_1, 0, horizontal_1, 0))

            feature_u_0_W =  nn.parallel.data_parallel(self.model_u_1_W, torch.cat([feature_d_1, feature_u_1_W], 1), self.gpu_ids)
            feature_u_0_W = torch.nn.functional.pad(feature_u_0_W, (vertical_0, 0, horizontal_0, 0))

            output_W =  nn.parallel.data_parallel(self.model_u_0_W, torch.cat([feature_d_0, feature_u_0_W], 1), self.gpu_ids)

        else:
            # downsampling features
            feature_d_0 = self.model_d_0(input)
            feature_d_1 = self.model_d_1(feature_d_0)
            feature_d_2 = self.model_d_2(feature_d_1)
            feature_d_3 = self.model_d_3(feature_d_2)
            feature_d_4 = self.model_d_4(feature_d_3)
            feature_d_inner = self.model_d_inner(feature_d_4)

            # upsampling transmission features ('back' is for the old name 'background')
            feature_u_4_back = self.model_u_inner_back(feature_d_inner)
            horizontal_4 = feature_d_4.shape[2] - feature_u_4_back.shape[2]
            vertical_4 = feature_d_4.shape[3] - feature_u_4_back.shape[3]
            feature_u_4_back = torch.nn.functional.pad(feature_u_4_back, (vertical_4, 0, horizontal_4, 0))

            feature_u_3_back = self.model_u_4_back(torch.cat([feature_d_4, feature_u_4_back], 1))
            horizontal_3 = feature_d_3.shape[2] - feature_u_3_back.shape[2]
            vertical_3 = feature_d_3.shape[3] - feature_u_3_back.shape[3]
            feature_u_3_back = torch.nn.functional.pad(feature_u_3_back, (vertical_3, 0, horizontal_3, 0))

            feature_u_2_back = self.model_u_3_back(torch.cat([feature_d_3, feature_u_3_back], 1))
            horizontal_2 = feature_d_2.shape[2] - feature_u_2_back.shape[2]
            vertical_2 = feature_d_2.shape[3] - feature_u_2_back.shape[3]
            feature_u_2_back = torch.nn.functional.pad(feature_u_2_back, (vertical_2, 0, horizontal_2, 0))

            feature_u_1_back = self.model_u_2_back(torch.cat([feature_d_2, feature_u_2_back], 1))
            horizontal_1 = feature_d_1.shape[2] - feature_u_1_back.shape[2]
            vertical_1 = feature_d_1.shape[3] - feature_u_1_back.shape[3]
            feature_u_1_back = torch.nn.functional.pad(feature_u_1_back, (vertical_1, 0, horizontal_1, 0))

            feature_u_0_back = self.model_u_1_back(torch.cat([feature_d_1, feature_u_1_back], 1))
            horizontal_0 = feature_d_0.shape[2] - feature_u_0_back.shape[2]
            vertical_0 = feature_d_0.shape[3] - feature_u_0_back.shape[3]
            feature_u_0_back = torch.nn.functional.pad(feature_u_0_back, (vertical_0, 0, horizontal_0, 0))

            output_trans = self.model_u_0_back(torch.cat([feature_d_0, feature_u_0_back], 1))

            # upsampling reflection features
            feature_u_4_refl = self.model_u_inner_refl(feature_d_inner)
            feature_u_4_refl = torch.nn.functional.pad(feature_u_4_refl, (vertical_4, 0, horizontal_4, 0))

            feature_u_3_refl = self.model_u_4_refl(torch.cat([feature_d_4, feature_u_4_refl], 1))
            feature_u_3_refl = torch.nn.functional.pad(feature_u_3_refl, (vertical_3, 0, horizontal_3, 0))

            feature_u_2_refl = self.model_u_3_refl(torch.cat([feature_d_3, feature_u_3_refl], 1))
            feature_u_2_refl = torch.nn.functional.pad(feature_u_2_refl, (vertical_2, 0, horizontal_2, 0))

            feature_u_1_refl = self.model_u_2_refl(torch.cat([feature_d_2, feature_u_2_refl], 1))
            feature_u_1_refl = torch.nn.functional.pad(feature_u_1_refl, (vertical_1, 0, horizontal_1, 0))

            feature_u_0_refl = self.model_u_1_refl(torch.cat([feature_d_1, feature_u_1_refl], 1))
            feature_u_0_refl = torch.nn.functional.pad(feature_u_0_refl, (vertical_0, 0, horizontal_0, 0))

            output_refl = self.model_u_0_refl(torch.cat([feature_d_0, feature_u_0_refl], 1))

            # upsampling W features
            feature_u_4_W = self.model_u_inner_W(feature_d_inner)
            feature_u_4_W = torch.nn.functional.pad(feature_u_4_W, (vertical_4, 0, horizontal_4, 0))

            feature_u_3_W = self.model_u_4_W(torch.cat([feature_d_4, feature_u_4_W], 1))
            feature_u_3_W = torch.nn.functional.pad(feature_u_3_W, (vertical_3, 0, horizontal_3, 0))

            feature_u_2_W = self.model_u_3_W(torch.cat([feature_d_3, feature_u_3_W], 1))
            feature_u_2_W = torch.nn.functional.pad(feature_u_2_W, (vertical_2, 0, horizontal_2, 0))

            feature_u_1_W = self.model_u_2_W(torch.cat([feature_d_2, feature_u_2_W], 1))
            feature_u_1_W = torch.nn.functional.pad(feature_u_1_W, (vertical_1, 0, horizontal_1, 0))

            feature_u_0_W = self.model_u_1_W(torch.cat([feature_d_1, feature_u_1_W], 1))
            feature_u_0_W = torch.nn.functional.pad(feature_u_0_W, (vertical_0, 0, horizontal_0, 0))

            output_W = self.model_u_0_W(torch.cat([feature_d_0, feature_u_0_W], 1))

        return output_trans, output_refl, output_W


class ImageGradient(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(ImageGradient, self).__init__()
        self.gpu_ids = gpu_ids

        a = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                       [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float(), requires_grad=False)

        b = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                       [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float(), requires_grad=False)

        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            G_x = nn.parallel.data_parallel(self.conv1, input, self.gpu_ids)
            G_y = nn.parallel.data_parallel(self.conv2, input, self.gpu_ids)
        else:
            G_x = conv1(input)
            G_y = conv2(input)
        
        return G_x, G_y