import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn as nn
 

class ReflectionRemovalModel(BaseModel):
    def name(self):
        return 'ReflectionRemovalModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # load/define networks

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.get_gradient = networks.ImageGradient(self.gpu_ids)

        if len(self.gpu_ids) > 0:
            self.get_gradient.cuda(self.gpu_ids[0])

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_C = input['C']
        if self.opt.phase == 'train':
            input_A = input['A']
            input_B = input['B']
            input_W = input['W']
        if len(self.gpu_ids) > 0:
            input_C = input_C.cuda(self.gpu_ids[0], async=True)
            if self.opt.phase == 'train':
                input_A = input_A.cuda(self.gpu_ids[0], async=True)
                input_B = input_B.cuda(self.gpu_ids[0], async=True)
                input_W = input_W.cuda(self.gpu_ids[0], async=True)
        self.input_C = input_C
        if self.opt.phase == 'train':
            self.input_A = input_A
            self.input_B = input_B
            self.input_W = input_W
        self.image_paths = input['C_path']

    def forward(self):
        self.real_C = self.input_C
        if self.opt.phase == 'train':
            self.real_reflection = self.input_A
            self.real_transmission = self.input_B
            self.real_W = self.input_W

    def test(self):
        real_C = self.input_C
        fake_transmission, fake_reflection, fake_W = self.netG(real_C)
        synthetic_C = fake_transmission * (1-fake_W) + fake_reflection * fake_W

        self.fake_transmission = fake_transmission.data
        self.fake_reflection = fake_reflection.data
        self.fake_W = fake_W.data
        self.synthetic_C = synthetic_C.data
        self.real_C = real_C.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        fake_transmission, fake_reflection, fake_W = self.netG(self.real_C)
        fake_transmission_grad_x, fake_transmission_grad_y = self.get_gradient(fake_transmission)
        real_transmission_grad_x, real_transmission_grad_y = self.get_gradient(self.real_transmission)
        synthetic_C = fake_transmission * (1-fake_W) + fake_reflection * fake_W
        loss_fake_transmission = self.criterionL1(fake_transmission, self.real_transmission) * 100
        loss_fake_reflection = self.criterionL1(fake_reflection, self.real_reflection) * 50
        loss_fake_transmission_grad = self.criterionL1(fake_transmission_grad_x, real_transmission_grad_x) * 50 \
                                      + self.criterionL1(fake_transmission_grad_y, real_transmission_grad_y) * 50

        loss_W = self.criterionL1(fake_W, self.real_W) * 100
        loss_IO = self.criterionL1(synthetic_C, self.real_C) * 50

        loss_G = loss_fake_transmission + loss_fake_reflection + loss_W + loss_IO \
               + loss_fake_transmission_grad

        loss_G.backward()

        self.fake_transmission = fake_transmission.data
        self.fake_reflection = fake_reflection.data
        self.fake_W = fake_W.data
        self.synthetic_C = synthetic_C.data

        self.loss_fake_transmission = loss_fake_transmission.item()
        self.loss_fake_reflection = loss_fake_reflection.item()
        self.loss_W = loss_W.item()
        self.loss_IO = loss_IO.item()
        self.loss_fake_transmission_grad = loss_fake_transmission_grad.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('loss_fake_transmission', self.loss_fake_transmission),
                                  ('loss_fake_reflection', self.loss_fake_reflection),
                                  ('loss_W', self.loss_W),
                                  ('loss_IO', self.loss_IO),
                                  ('loss_fake_transmission_grad', self.loss_fake_transmission_grad)])
        return ret_errors

    def get_current_visuals_train(self):
#        real_transmission = util.tensor2im(self.real_transmission)
#        real_reflection = util.tensor2im(self.real_reflection)
        real_C = util.tensor2im(self.real_C)
        fake_transmission = util.tensor2im(self.fake_transmission)
        fake_reflection = util.tensor2im(self.fake_reflection)
#        synthetic_C = util.tensor2im(self.synthetic_C)

        ret_visuals = OrderedDict([('fake_transmission', fake_transmission), ('fake_reflection', fake_reflection), ('real_C', real_C)])
                                #    ('real_transmission', real_transmission), ('real_reflection', real_reflection), ('synthetic_C', synthetic_C)
        return ret_visuals

    def get_current_visuals_test(self):
        fake_transmission = util.tensor2im(self.fake_transmission)
#        fake_reflection = util.tensor2im(self.fake_reflection)
        real_C = util.tensor2im(self.real_C)
        synthetic_C = util.tensor2im(self.synthetic_C)

        ret_visuals = OrderedDict([('fake_transmission', fake_transmission), ('real_C', real_C), ('synthetic_C', synthetic_C)])
#                                    ('fake_reflection', fake_reflection)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
