import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from model.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn as nn


class ReflectionSynthesisModel(BaseModel):
    def name(self):
        return 'ReflectionSynthesisModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # load/define networks

        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.mix_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_A_origin = input['A_origin']
        input_B = input['B']
        One = torch.ones(input_A.shape)
            
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            One = One.cuda(self.gpu_ids[0], async=True)
                        
        self.input_A = input_A
        self.input_A_origin = input_A_origin
        self.input_B = input_B
        self.One = One
        if self.opt.phase == 'train':
            input_C = input['C']
            input_C = input_C.cuda(self.gpu_ids[0], async=True)
            self.input_C = input_C
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = self.input_A
        self.real_A_origin = self.input_A_origin
        self.real_B = self.input_B
        if self.opt.phase == 'train':
            self.real_C = self.input_C

    def test(self):
        real_A = self.input_A
        real_A_origin = self.input_A_origin
        real_B = self.input_B
        concat_AB = torch.cat((real_B, real_A), dim=1)
        W_A_reflection = self.netG(concat_AB)
        W_A_reflection_revise = self.One - W_A_reflection
        mix_AB = W_A_reflection * real_A + W_A_reflection_revise * real_B

        self.real_A = real_A.data
        self.real_A_origin = real_A_origin.data
        self.real_B = real_B.data
        self.W_A_reflection = W_A_reflection.data
        self.mix_AB = mix_AB.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        mix_AB = self.mix_AB_pool.query(self.mix_AB)
        loss_D = self.backward_D_basic(self.netD, self.real_C, mix_AB)
        self.loss_D = loss_D.item()

    def backward_G(self):

        # GAN loss D(G(concat_AB))
        reflection = self.real_A
        transmission = self.real_B
        concat_AB = torch.cat((transmission, reflection), dim=1)
        W = self.netG(concat_AB)
        W_revise = self.One - W
        mix_AB = W_revise * transmission + W * reflection
        pred_fake = self.netD(mix_AB)
        loss_GAN = self.criterionGAN(pred_fake, True)

        # for smoothness loss
        smooth_y_W = self.criterionL2(W[:, :, 1:, :], W.detach()[:, :, :-1, :])
        smooth_x_W = self.criterionL2(W[:, :, :, 1:], W.detach()[:, :, :, :-1])
        loss_Smooth_W = smooth_y_W + smooth_x_W

        loss_G = loss_GAN + loss_Smooth_W * 10

        loss_G.backward()

        self.mix_AB = mix_AB.data
        self.reflection = reflection.data
        self.transmission = transmission.data
        self.W = W.data
        self.W_revise = W_revise.data

        self.loss_GAN = loss_GAN.item()
        self.loss_Smooth_W = loss_Smooth_W.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('loss_GAN', self.loss_GAN),
                                  ('loss_Smooth_W', self.loss_Smooth_W),
                                  ('loss_D', self.loss_D)])
        return ret_errors

    def get_current_visuals_train(self):
        reflection = util.tensor2im(self.reflection)
        transmission = util.tensor2im(self.transmission)
        real_C = util.tensor2im(self.input_C)
        mix_AB = util.tensor2im(self.mix_AB)

        ret_visuals = OrderedDict([('reflection', reflection),('transmission', transmission),
                                   ('real_C', real_C), ('mix_AB', mix_AB)])
        return ret_visuals

    def get_current_visuals_test(self):
        real_A = util.tensor2im(self.real_A)
        real_A_origin = util.tensor2im(self.real_A_origin)
        real_B = util.tensor2im(self.real_B)
        mix_AB = util.tensor2im(self.mix_AB)
        ret_visuals = OrderedDict([('reflection', real_A), ('transmission', real_B), ('reflection_origin', real_A_origin),
                                   ('mix_AB', mix_AB)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)