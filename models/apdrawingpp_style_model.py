import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os
import math

W = 11
aa = int(math.floor(512./W))
res = 512 - W*aa


def padpart(A,part,centers,opt,device):
    IMAGE_SIZE = opt.fineSize
    bs,nc,_,_ = A.shape
    ratio = IMAGE_SIZE / 256
    NOSE_W = opt.NOSE_W * ratio
    NOSE_H = opt.NOSE_H * ratio
    EYE_W = opt.EYE_W * ratio
    EYE_H = opt.EYE_H * ratio
    MOUTH_W = opt.MOUTH_W * ratio
    MOUTH_H = opt.MOUTH_H * ratio
    A_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(device)
    padvalue = -1 # black
    for i in range(bs):
        center = centers[i]
        if part == 'nose':
            A_p[i] = torch.nn.ConstantPad2d((center[2,0] - NOSE_W / 2, IMAGE_SIZE - (center[2,0]+NOSE_W/2), center[2,1] - NOSE_H / 2, IMAGE_SIZE - (center[2,1]+NOSE_H/2)),padvalue)(A[i])
        elif part == 'eyel':
            A_p[i] = torch.nn.ConstantPad2d((center[0,0] - EYE_W / 2, IMAGE_SIZE - (center[0,0]+EYE_W/2), center[0,1] - EYE_H / 2, IMAGE_SIZE - (center[0,1]+EYE_H/2)),padvalue)(A[i])
        elif part == 'eyer':
            A_p[i] = torch.nn.ConstantPad2d((center[1,0] - EYE_W / 2, IMAGE_SIZE - (center[1,0]+EYE_W/2), center[1,1] - EYE_H / 2, IMAGE_SIZE - (center[1,1]+EYE_H/2)),padvalue)(A[i])
        elif part == 'mouth':
            A_p[i] = torch.nn.ConstantPad2d((center[3,0] - MOUTH_W / 2, IMAGE_SIZE - (center[3,0]+MOUTH_W/2), center[3,1] - MOUTH_H / 2, IMAGE_SIZE - (center[3,1]+MOUTH_H/2)),padvalue)(A[i])
    return A_p

import numpy as np
def nonlinearDt(dt,type='atan',xmax=torch.Tensor([10.0])):#dt in [0,1], first multiply xmax(>1), then remap to [0,1]
    if type == 'atan':
        nldt = torch.atan(dt*xmax) / torch.atan(xmax)
    elif type == 'sigmoid':
        nldt = (torch.sigmoid(dt*xmax)-0.5) / (torch.sigmoid(xmax)-0.5)
    elif type == 'tanh':
        nldt = torch.tanh(dt*xmax) / torch.tanh(xmax)
    elif type == 'pow':
        nldt = torch.pow(dt*xmax,2) / torch.pow(xmax,2)
    elif type == 'exp':
        if xmax.item()>1:
            xmax = xmax / 3
        nldt = (torch.exp(dt*xmax)-1) / (torch.exp(xmax)-1)
    #print("remap dt:", type, xmax.item())
    return nldt

class APDrawingPPStyleModel(BaseModel):
    def name(self):
        return 'APDrawingPPStyleModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')# no_lsgan=True, use_lsgan=False
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(auxiliary_root='auxiliaryeye2o')
        parser.set_defaults(use_local=True, hair_local=True, bg_local=True)
        parser.set_defaults(discriminator_local=True, gan_loss_strategy=2)
        parser.set_defaults(chamfer_loss=True, dt_nonlinear='exp', lambda_chamfer=0.35, lambda_chamfer2=0.35)
        parser.set_defaults(nose_ae=True, others_ae=True, compactmask=True, MOUTH_H=56)
        parser.set_defaults(soft_border=1, batch_size=1, save_epoch_freq=25)
        parser.add_argument('--nnG_hairc', type=int, default=6, help='nnG for hair classifier')
        parser.add_argument('--use_resnet', action='store_true', help='use resnet for generator')
        parser.add_argument('--regarch', type=int, default=4, help='architecture for netRegressor')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_local', type=float, default=25.0, help='weight for Local loss')
            parser.set_defaults(netG_dt='unet_512')
            parser.set_defaults(netG_line='unet_512')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if self.isTrain and self.opt.no_l1_loss:
            self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        if self.isTrain and self.opt.use_local and not self.opt.no_G_local_loss:
            self.loss_names.append('G_local')
            self.loss_names.append('G_hair_local')
            self.loss_names.append('G_bg_local')
        if self.isTrain and self.opt.discriminator_local:
            self.loss_names.append('D_real_local')
            self.loss_names.append('D_fake_local')
            self.loss_names.append('G_GAN_local')
        if self.isTrain and self.opt.chamfer_loss:
            self.loss_names.append('G_chamfer')
            self.loss_names.append('G_chamfer2')
        if self.isTrain and self.opt.continuity_loss:
            self.loss_names.append('G_continuity')
        self.loss_names.append('G')
        print('loss_names', self.loss_names)
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.opt.use_local:
            self.visual_names += ['fake_B0', 'fake_B1']
            self.visual_names += ['fake_B_hair', 'real_B_hair', 'real_A_hair']
            self.visual_names += ['fake_B_bg', 'real_B_bg', 'real_A_bg']
            if self.opt.region_enm in [0,1]:
                if self.opt.nose_ae:
                    self.visual_names += ['fake_B_nose_v','fake_B_nose_v1','fake_B_nose_v2','cmask1no']
                if self.opt.others_ae:
                    self.visual_names += ['fake_B_eyel_v','fake_B_eyel_v1','fake_B_eyel_v2','cmask1el']
                    self.visual_names += ['fake_B_eyer_v','fake_B_eyer_v1','fake_B_eyer_v2','cmask1er']
                    self.visual_names += ['fake_B_mouth_v','fake_B_mouth_v1','fake_B_mouth_v2','cmask1mo']
            elif self.opt.region_enm in [2]:
                self.visual_names += ['fake_B_nose','fake_B_eyel','fake_B_eyer','fake_B_mouth']
        if self.isTrain and self.opt.chamfer_loss:
            self.visual_names += ['dt1', 'dt2']
            self.visual_names += ['dt1gt', 'dt2gt']
        if self.isTrain and self.opt.soft_border:
            self.visual_names += ['mask']
        if not self.isTrain and self.opt.save2:
            self.visual_names = ['real_A', 'fake_B']
        print('visuals', self.visual_names)
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.auxiliary_model_names = []
        if self.isTrain:
            self.model_names = ['G', 'D']
            if self.opt.discriminator_local:
                self.model_names += ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            # auxiliary nets for loss calculation
            if self.opt.chamfer_loss:
                self.auxiliary_model_names += ['DT1', 'DT2']
                self.auxiliary_model_names += ['Line1', 'Line2']
            if self.opt.continuity_loss:
                self.auxiliary_model_names += ['Regressor']
        else:  # during test time, only load Gs
            self.model_names = ['G']
            if self.opt.test_continuity_loss:
                self.auxiliary_model_names += ['Regressor']
        if self.opt.use_local:
            self.model_names += ['GLEyel','GLEyer','GLNose','GLMouth','GLHair','GLBG','GCombine']
            self.auxiliary_model_names += ['CLm','CLh']
            # auxiliary nets for local output refinement
            if self.opt.nose_ae:
                self.auxiliary_model_names += ['AE']
            if self.opt.others_ae:
                self.auxiliary_model_names += ['AEel','AEer','AEmowhite','AEmoblack']
        print('model_names', self.model_names)
        print('auxiliary_model_names', self.auxiliary_model_names)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.nnG)
        print('netG', opt.netG)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            print('netD', opt.netD, opt.n_layers_D)
            if self.opt.discriminator_local:
                self.netDLEyel = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLEyer = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLNose = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLMouth = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLHair = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLBG = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                
        
        if self.opt.use_local:
            netlocal1 = 'partunet' if self.opt.use_resnet == 0 else 'resnet_nblocks'
            netlocal2 = 'partunet2' if self.opt.use_resnet == 0 else 'resnet_6blocks'
            netlocal2_style = 'partunet2style' if self.opt.use_resnet == 0 else 'resnet_style2_6blocks'
            self.netGLEyel = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, netlocal1, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, nnG=3)
            self.netGLEyer = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, netlocal1, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, nnG=3)
            self.netGLNose = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, netlocal1, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, nnG=3)
            self.netGLMouth = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, netlocal1, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, nnG=3)
            self.netGLHair = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, netlocal2_style, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, nnG=4,
                                      extra_channel=3)
            self.netGLBG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, netlocal2, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, nnG=4)
            # by default combiner_type is combiner, which uses resnet
            print('combiner_type', self.opt.combiner_type)
            self.netGCombine = networks.define_G(2*opt.output_nc, opt.output_nc, opt.ngf, self.opt.combiner_type, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 2)
            # auxiliary classifiers for mouth and hair
            ratio = self.opt.fineSize / 256
            self.MOUTH_H = int(self.opt.MOUTH_H * ratio)
            self.MOUTH_W = int(self.opt.MOUTH_W * ratio)
            self.netCLm = networks.define_G(opt.input_nc, 2, opt.ngf, 'classifier', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      nnG = 3, ae_h = self.MOUTH_H, ae_w = self.MOUTH_W)
            self.netCLh = networks.define_G(opt.input_nc, 3, opt.ngf, 'classifier', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      nnG = opt.nnG_hairc, ae_h = opt.fineSize, ae_w = opt.fineSize)
        

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            if not self.opt.use_local:
                print('G_params 1 components')
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                G_params = list(self.netG.parameters()) + list(self.netGLEyel.parameters()) + list(self.netGLEyer.parameters()) + list(self.netGLNose.parameters()) + list(self.netGLMouth.parameters()) + list(self.netGCombine.parameters()) + list(self.netGLHair.parameters()) + list(self.netGLBG.parameters())
                print('G_params 8 components')
                self.optimizer_G = torch.optim.Adam(G_params,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            if not self.opt.discriminator_local:
                print('D_params 1 components')
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            else:#self.opt.discriminator_local == True
                D_params = list(self.netD.parameters()) + list(self.netDLEyel.parameters()) +list(self.netDLEyer.parameters()) + list(self.netDLNose.parameters()) + list(self.netDLMouth.parameters()) + list(self.netDLHair.parameters()) + list(self.netDLBG.parameters())
                print('D_params 7 components')
                self.optimizer_D = torch.optim.Adam(D_params,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        # ==================================auxiliary nets (loaded, parameters fixed)=============================
        if self.opt.use_local and self.opt.nose_ae:
            ratio = self.opt.fineSize / 256
            NOSE_H = self.opt.NOSE_H * ratio
            NOSE_W = self.opt.NOSE_W * ratio
            self.netAE = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, self.opt.nose_ae_net, 'batch',
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                       latent_dim=self.opt.ae_latentno, ae_h=NOSE_H, ae_w=NOSE_W)
            self.set_requires_grad(self.netAE, False)            
        if self.opt.use_local and self.opt.others_ae:
            ratio = self.opt.fineSize / 256
            EYE_H = self.opt.EYE_H * ratio
            EYE_W = self.opt.EYE_W * ratio
            MOUTH_H = self.opt.MOUTH_H * ratio
            MOUTH_W = self.opt.MOUTH_W * ratio
            self.netAEel = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, self.opt.nose_ae_net, 'batch',
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      latent_dim=self.opt.ae_latenteye, ae_h=EYE_H, ae_w=EYE_W)
            self.netAEer = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, self.opt.nose_ae_net, 'batch',
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      latent_dim=self.opt.ae_latenteye, ae_h=EYE_H, ae_w=EYE_W)
            self.netAEmowhite = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, self.opt.nose_ae_net, 'batch',
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      latent_dim=self.opt.ae_latentmo, ae_h=MOUTH_H, ae_w=MOUTH_W)
            self.netAEmoblack = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, self.opt.nose_ae_net, 'batch',
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      latent_dim=self.opt.ae_latentmo, ae_h=MOUTH_H, ae_w=MOUTH_W)
            self.set_requires_grad(self.netAEel, False)
            self.set_requires_grad(self.netAEer, False)
            self.set_requires_grad(self.netAEmowhite, False)
            self.set_requires_grad(self.netAEmoblack, False)
            
        
        if self.isTrain and self.opt.continuity_loss:
            self.nc = 1
            self.netRegressor = networks.define_G(self.nc, 1, opt.ngf, 'regressor', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids_p,
                                      nnG = opt.regarch)
            self.set_requires_grad(self.netRegressor, False)

        if self.isTrain and self.opt.chamfer_loss:
            self.nc = 1
            self.netDT1 = networks.define_G(self.nc, self.nc, opt.ngf, opt.netG_dt, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids_p)
            self.netDT2 = networks.define_G(self.nc, self.nc, opt.ngf, opt.netG_dt, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids_p)
            self.set_requires_grad(self.netDT1, False)
            self.set_requires_grad(self.netDT2, False)
            self.netLine1 = networks.define_G(self.nc, self.nc, opt.ngf, opt.netG_line, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids_p)
            self.netLine2 = networks.define_G(self.nc, self.nc, opt.ngf, opt.netG_line, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids_p)
            self.set_requires_grad(self.netLine1, False)
            self.set_requires_grad(self.netLine2, False)
        
        # ==================================for test (nets loaded, parameters fixed)=============================
        if  not self.isTrain and self.opt.test_continuity_loss:
            self.nc = 1
            self.netRegressor = networks.define_G(self.nc, 1, opt.ngf, 'regressor', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      nnG = opt.regarch)
            self.set_requires_grad(self.netRegressor, False)
        

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.batch_size = len(self.image_paths)
        if self.opt.use_local:
            self.real_A_eyel = input['eyel_A'].to(self.device)
            self.real_A_eyer = input['eyer_A'].to(self.device)
            self.real_A_nose = input['nose_A'].to(self.device)
            self.real_A_mouth = input['mouth_A'].to(self.device)
            self.real_B_eyel = input['eyel_B'].to(self.device)
            self.real_B_eyer = input['eyer_B'].to(self.device)
            self.real_B_nose = input['nose_B'].to(self.device)
            self.real_B_mouth = input['mouth_B'].to(self.device)
            if self.opt.region_enm in [0,1]:
                self.center = input['center']
            if self.opt.soft_border:
                self.softel = input['soft_eyel_mask'].to(self.device)
                self.softer = input['soft_eyer_mask'].to(self.device)
                self.softno = input['soft_nose_mask'].to(self.device)
                self.softmo = input['soft_mouth_mask'].to(self.device)
            if self.opt.compactmask:
                self.cmask = input['cmask'].to(self.device)
                self.cmask1 = self.cmask*2-1#[0,1]->[-1,1]
                self.cmaskel = input['cmaskel'].to(self.device)
                self.cmask1el = self.cmaskel*2-1
                self.cmasker = input['cmasker'].to(self.device)
                self.cmask1er = self.cmasker*2-1
                self.cmaskmo = input['cmaskmo'].to(self.device)
                self.cmask1mo = self.cmaskmo*2-1
            self.real_A_hair = input['hair_A'].to(self.device)
            self.real_B_hair = input['hair_B'].to(self.device)
            self.mask = input['mask'].to(self.device) # mask for non-eyes,nose,mouth
            self.mask2 = input['mask2'].to(self.device) # mask for non-bg
            self.real_A_bg = input['bg_A'].to(self.device)
            self.real_B_bg = input['bg_B'].to(self.device)
        if (self.isTrain and self.opt.chamfer_loss):
            self.dt1gt = input['dt1gt'].to(self.device)
            self.dt2gt = input['dt2gt'].to(self.device)
        if self.isTrain and self.opt.emphasis_conti_face:
            self.face_mask = input['face_mask'].cuda(self.gpu_ids_p[0])
        
    def getonehot(self,outputs,classes):
        [maxv,index] = torch.max(outputs,1)
        y = torch.unsqueeze(index,1)
        onehot = torch.FloatTensor(self.batch_size,classes).to(self.device)
        onehot.zero_()
        onehot.scatter_(1,y,1)
        return onehot

    def forward(self):
        if not self.opt.use_local:
            self.fake_B = self.netG(self.real_A)
        else:
            self.fake_B0 = self.netG(self.real_A)
            # EYES, MOUTH
            outputs1 = self.netCLm(self.real_A_mouth)
            onehot1 = self.getonehot(outputs1,2)

            if not self.opt.others_ae:
                fake_B_eyel = self.netGLEyel(self.real_A_eyel)
                fake_B_eyer = self.netGLEyer(self.real_A_eyer)
                fake_B_mouth = self.netGLMouth(self.real_A_mouth)
            else: # use AE that only constains compact region, need cmask!
                self.fake_B_eyel1 = self.netGLEyel(self.real_A_eyel)
                self.fake_B_eyer1 = self.netGLEyer(self.real_A_eyer)
                self.fake_B_mouth1 = self.netGLMouth(self.real_A_mouth)
                self.fake_B_eyel2,_ = self.netAEel(self.fake_B_eyel1)
                self.fake_B_eyer2,_ = self.netAEer(self.fake_B_eyer1)
                # USE 2 AEs
                self.fake_B_mouth2 = torch.FloatTensor(self.batch_size,self.opt.output_nc,self.MOUTH_H,self.MOUTH_W).to(self.device)
                for i in range(self.batch_size):
                    if onehot1[i][0] == 1:
                        self.fake_B_mouth2[i],_ = self.netAEmowhite(self.fake_B_mouth1[i].unsqueeze(0))
                        #print('AEmowhite')
                    elif onehot1[i][1] == 1:
                        self.fake_B_mouth2[i],_ = self.netAEmoblack(self.fake_B_mouth1[i].unsqueeze(0))
                        #print('AEmoblack')
                fake_B_eyel = self.add_with_mask(self.fake_B_eyel2,self.fake_B_eyel1,self.cmaskel)
                fake_B_eyer = self.add_with_mask(self.fake_B_eyer2,self.fake_B_eyer1,self.cmasker)
                fake_B_mouth = self.add_with_mask(self.fake_B_mouth2,self.fake_B_mouth1,self.cmaskmo)
            # NOSE
            if not self.opt.nose_ae:
                fake_B_nose = self.netGLNose(self.real_A_nose)
            else: # use AE that only constains compact region, need cmask!
                self.fake_B_nose1 = self.netGLNose(self.real_A_nose)
                self.fake_B_nose2,_ = self.netAE(self.fake_B_nose1)
                fake_B_nose = self.add_with_mask(self.fake_B_nose2,self.fake_B_nose1,self.cmask)

            # for visuals and later local loss
            if self.opt.region_enm in [0,1]:
                self.fake_B_nose = fake_B_nose
                self.fake_B_eyel = fake_B_eyel
                self.fake_B_eyer = fake_B_eyer
                self.fake_B_mouth = fake_B_mouth
                # for soft border of 4 rectangle facial feature
                if self.opt.region_enm == 0 and self.opt.soft_border:
                    self.fake_B_nose = self.masked(fake_B_nose, self.softno)
                    self.fake_B_eyel = self.masked(fake_B_eyel, self.softel)
                    self.fake_B_eyer = self.masked(fake_B_eyer, self.softer)
                    self.fake_B_mouth = self.masked(fake_B_mouth, self.softmo)
            elif self.opt.region_enm in [2]: # need to multiply cmask
                self.fake_B_nose = self.masked(fake_B_nose,self.cmask)
                self.fake_B_eyel = self.masked(fake_B_eyel,self.cmaskel)
                self.fake_B_eyer = self.masked(fake_B_eyer,self.cmasker)
                self.fake_B_mouth = self.masked(fake_B_mouth,self.cmaskmo)
            
            # HAIR, BG AND PARTCOMBINE
            outputs2 = self.netCLh(self.real_A_hair)
            onehot2 = self.getonehot(outputs2,3)

            if not self.isTrain:
                opt = self.opt
                if opt.imagefolder == 'images':
                    file_name = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'styleonehot.txt')
                else:
                    file_name = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), opt.imagefolder, 'styleonehot.txt')
                message = '%s [%d %d] [%d %d %d]' % (self.image_paths[0], onehot1[0][0], onehot1[0][1], 
                onehot2[0][0], onehot2[0][1], onehot2[0][2])
                with open(file_name, 'a+') as s_file:
                    s_file.write(message)
                    s_file.write('\n')

            fake_B_hair = self.netGLHair(self.real_A_hair,onehot2)
            fake_B_bg = self.netGLBG(self.real_A_bg)
            self.fake_B_hair = self.masked(fake_B_hair,self.mask*self.mask2)
            self.fake_B_bg = self.masked(fake_B_bg,self.inverse_mask(self.mask2))
            if not self.opt.compactmask:
                self.fake_B1 = self.partCombiner2_bg(fake_B_eyel,fake_B_eyer,fake_B_nose,fake_B_mouth,fake_B_hair,fake_B_bg,self.mask*self.mask2,self.inverse_mask(self.mask2),self.opt.comb_op)
            else:
                self.fake_B1 = self.partCombiner2_bg(fake_B_eyel,fake_B_eyer,fake_B_nose,fake_B_mouth,fake_B_hair,fake_B_bg,self.mask*self.mask2,self.inverse_mask(self.mask2),self.opt.comb_op,self.opt.region_enm,self.cmaskel,self.cmasker,self.cmask,self.cmaskmo)
            
            self.fake_B = self.netGCombine(torch.cat([self.fake_B0,self.fake_B1],1))

            # for AE visuals
            if self.opt.region_enm in [0,1]:
                if self.opt.nose_ae:
                    self.fake_B_nose_v = padpart(self.fake_B_nose, 'nose', self.center, self.opt, self.device)
                    self.fake_B_nose_v1 = padpart(self.fake_B_nose1, 'nose', self.center, self.opt, self.device)
                    self.fake_B_nose_v2 = padpart(self.fake_B_nose2, 'nose', self.center, self.opt, self.device)
                    self.cmask1no = padpart(self.cmask1, 'nose', self.center, self.opt, self.device)
                if self.opt.others_ae:
                    self.fake_B_eyel_v = padpart(self.fake_B_eyel, 'eyel', self.center, self.opt, self.device)
                    self.fake_B_eyel_v1 = padpart(self.fake_B_eyel1, 'eyel', self.center, self.opt, self.device)
                    self.fake_B_eyel_v2 = padpart(self.fake_B_eyel2, 'eyel', self.center, self.opt, self.device)
                    self.cmask1el = padpart(self.cmask1el, 'eyel', self.center, self.opt, self.device)
                    self.fake_B_eyer_v = padpart(self.fake_B_eyer, 'eyer', self.center, self.opt, self.device)
                    self.fake_B_eyer_v1 = padpart(self.fake_B_eyer1, 'eyer', self.center, self.opt, self.device)
                    self.fake_B_eyer_v2 = padpart(self.fake_B_eyer2, 'eyer', self.center, self.opt, self.device)
                    self.cmask1er = padpart(self.cmask1er, 'eyer', self.center, self.opt, self.device)
                    self.fake_B_mouth_v = padpart(self.fake_B_mouth, 'mouth', self.center, self.opt, self.device)
                    self.fake_B_mouth_v1 = padpart(self.fake_B_mouth1, 'mouth', self.center, self.opt, self.device)
                    self.fake_B_mouth_v2 = padpart(self.fake_B_mouth2, 'mouth', self.center, self.opt, self.device)
                    self.cmask1mo = padpart(self.cmask1mo, 'mouth', self.center, self.opt, self.device)
            
            if not self.isTrain and self.opt.test_continuity_loss:
                self.ContinuityForTest(real=1)
    
        
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        #print('fake_AB', fake_AB.shape) # (1,4,512,512)
        pred_fake = self.netD(fake_AB.detach())# by detach, not affect G's gradient
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        if self.opt.discriminator_local:
            fake_AB_parts = self.getLocalParts(fake_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_D_fake_local = 0
            for i in range(len(fake_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_fake_tmp = net(fake_AB_parts[i].detach())
                addw = self.getaddw(local_names[i])
                self.loss_D_fake_local = self.loss_D_fake_local + self.criterionGAN(pred_fake_tmp, False) * addw
            self.loss_D_fake = self.loss_D_fake + self.loss_D_fake_local

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        if self.opt.discriminator_local:
            real_AB_parts = self.getLocalParts(real_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_D_real_local = 0
            for i in range(len(real_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_real_tmp = net(real_AB_parts[i])
                addw = self.getaddw(local_names[i])
                self.loss_D_real_local = self.loss_D_real_local + self.criterionGAN(pred_real_tmp, True) * addw
            self.loss_D_real = self.loss_D_real + self.loss_D_real_local

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB) # (1,4,512,512)->(1,1,30,30)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        if self.opt.discriminator_local:
            fake_AB_parts = self.getLocalParts(fake_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_G_GAN_local = 0 # G_GAN_local is then added into G_GAN
            for i in range(len(fake_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_fake_tmp = net(fake_AB_parts[i])
                addw = self.getaddw(local_names[i])
                self.loss_G_GAN_local = self.loss_G_GAN_local + self.criterionGAN(pred_fake_tmp, True) * addw
            if self.opt.gan_loss_strategy == 1:
                self.loss_G_GAN = (self.loss_G_GAN + self.loss_G_GAN_local) / (len(fake_AB_parts) + 1)
            elif self.opt.gan_loss_strategy == 2:
                self.loss_G_GAN_local = self.loss_G_GAN_local * 0.25
                self.loss_G_GAN = self.loss_G_GAN + self.loss_G_GAN_local

        # Second, G(A) = B
        if not self.opt.no_l1_loss:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        if self.opt.use_local and not self.opt.no_G_local_loss:
            local_names = ['eyel','eyer','nose','mouth']
            self.loss_G_local = 0
            for i in range(len(local_names)):
                fakeblocal = getattr(self, 'fake_B_' + local_names[i])
                realblocal = getattr(self, 'real_B_' + local_names[i])
                addw = self.getaddw(local_names[i])
                self.loss_G_local = self.loss_G_local + self.criterionL1(fakeblocal,realblocal) * self.opt.lambda_local * addw
            self.loss_G_hair_local = self.criterionL1(self.fake_B_hair, self.real_B_hair) * self.opt.lambda_local * self.opt.addw_hair
            self.loss_G_bg_local = self.criterionL1(self.fake_B_bg, self.real_B_bg) * self.opt.lambda_local * self.opt.addw_bg

        # Third, chamfer matching (assume chamfer_2way and chamfer_only_line is true)
        if self.opt.chamfer_loss:
            if self.fake_B.shape[1] == 3:
                tmp = self.fake_B[:,0,...]*0.299+self.fake_B[:,1,...]*0.587+self.fake_B[:,2,...]*0.114
                fake_B_gray = tmp.unsqueeze(1)
            else:
                fake_B_gray = self.fake_B
            if self.real_B.shape[1] == 3:
                tmp = self.real_B[:,0,...]*0.299+self.real_B[:,1,...]*0.587+self.real_B[:,2,...]*0.114
                real_B_gray = tmp.unsqueeze(1)
            else:
                real_B_gray = self.real_B
            
            gpu_p = self.opt.gpu_ids_p[0]
            gpu = self.opt.gpu_ids[0]
            if gpu_p != gpu:
                fake_B_gray = fake_B_gray.cuda(gpu_p)
                real_B_gray = real_B_gray.cuda(gpu_p)

            # d_CM(a_i,G(p_i))
            self.dt1 = self.netDT1(fake_B_gray)
            self.dt2 = self.netDT2(fake_B_gray)
            dt1 = self.dt1/2.0+0.5#[-1,1]->[0,1]
            dt2 = self.dt2/2.0+0.5
            if self.opt.dt_nonlinear != '':
                dt_xmax = torch.Tensor([self.opt.dt_xmax]).cuda(gpu_p)
                dt1 = nonlinearDt(dt1, self.opt.dt_nonlinear, dt_xmax)
                dt2 = nonlinearDt(dt2, self.opt.dt_nonlinear, dt_xmax)
                #print('dt1dt2',torch.min(dt1).item(),torch.max(dt1).item(),torch.min(dt2).item(),torch.max(dt2).item())
            
            bs = real_B_gray.shape[0]
            real_B_gray_line1 = self.netLine1(real_B_gray)
            real_B_gray_line2 = self.netLine2(real_B_gray)
            self.loss_G_chamfer = (dt1[(real_B_gray<0)&(real_B_gray_line1<0)].sum() + dt2[(real_B_gray>=0)&(real_B_gray_line2>=0)].sum()) / bs * self.opt.lambda_chamfer
            if gpu_p != gpu:
                self.loss_G_chamfer = self.loss_G_chamfer.cuda(gpu) 

            # d_CM(G(p_i),a_i)
            if gpu_p != gpu:
                dt1gt = self.dt1gt.cuda(gpu_p)
                dt2gt = self.dt2gt.cuda(gpu_p)
            else:
                dt1gt = self.dt1gt
                dt2gt = self.dt2gt
            if self.opt.dt_nonlinear != '':
                dt1gt = nonlinearDt(dt1gt, self.opt.dt_nonlinear, dt_xmax)
                dt2gt = nonlinearDt(dt2gt, self.opt.dt_nonlinear, dt_xmax)
                #print('dt1gtdt2gt',torch.min(dt1gt).item(),torch.max(dt1gt).item(),torch.min(dt2gt).item(),torch.max(dt2gt).item())
            self.dt1gt = (self.dt1gt-0.5)*2
            self.dt2gt = (self.dt2gt-0.5)*2

            fake_B_gray_line1 = self.netLine1(fake_B_gray)
            fake_B_gray_line2 = self.netLine2(fake_B_gray)
            self.loss_G_chamfer2 = (dt1gt[(fake_B_gray<0)&(fake_B_gray_line1<0)].sum() + dt2gt[(fake_B_gray>=0)&(fake_B_gray_line2>=0)].sum()) / bs * self.opt.lambda_chamfer2
            if gpu_p != gpu:
                self.loss_G_chamfer2 = self.loss_G_chamfer2.cuda(gpu)

        # Fourth, line continuity loss, constrained on synthesized drawing
        if self.opt.continuity_loss:
            # Patch-based
            self.get_patches()
            self.outputs = self.netRegressor(self.fake_B_patches)
            if not self.opt.emphasis_conti_face:
                self.loss_G_continuity = (1.0-torch.mean(self.outputs)).cuda(gpu) * self.opt.lambda_continuity
            else:
                self.loss_G_continuity = torch.mean((1.0-self.outputs)*self.conti_weights).cuda(gpu) * self.opt.lambda_continuity



        self.loss_G = self.loss_G_GAN
        if 'G_L1' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_L1
        if 'G_local' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_local
        if 'G_hair_local' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_hair_local
        if 'G_bg_local' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_bg_local
        if 'G_chamfer' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_chamfer
        if 'G_chamfer2' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_chamfer2
        if 'G_continuity' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_continuity

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        
        if self.opt.discriminator_local:
            self.set_requires_grad(self.netDLEyel, True)
            self.set_requires_grad(self.netDLEyer, True)
            self.set_requires_grad(self.netDLNose, True)
            self.set_requires_grad(self.netDLMouth, True)
            self.set_requires_grad(self.netDLHair, True)
            self.set_requires_grad(self.netDLBG, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        if self.opt.discriminator_local:
            self.set_requires_grad(self.netDLEyel, False)
            self.set_requires_grad(self.netDLEyer, False)
            self.set_requires_grad(self.netDLNose, False)
            self.set_requires_grad(self.netDLMouth, False)
            self.set_requires_grad(self.netDLHair, False)
            self.set_requires_grad(self.netDLBG, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_patches(self):
        gpu_p = self.opt.gpu_ids_p[0]
        gpu = self.opt.gpu_ids[0]
        if gpu_p != gpu:
            self.fake_B = self.fake_B.cuda(gpu_p)
        # [1,1,512,512]->[bs,1,11,11]
        patches = []
        if self.isTrain and self.opt.emphasis_conti_face:
            weights = []
            W2 = int(W/2)
        t = np.random.randint(res,size=2)
        for i in range(aa):
            for j in range(aa):
                p = self.fake_B[:,:,t[0]+i*W:t[0]+(i+1)*W,t[1]+j*W:t[1]+(j+1)*W]
                whitenum = torch.sum(p>=0.0)
                #if whitenum < 5 or whitenum > W*W-5:
                if whitenum < 1 or whitenum > W*W-1:
                    continue
                patches.append(p)
                if self.isTrain and self.opt.emphasis_conti_face:
                    weights.append(self.face_mask[:,:,t[0]+i*W+W2,t[1]+j*W+W2])
        self.fake_B_patches = torch.cat(patches, dim=0)
        if self.isTrain and self.opt.emphasis_conti_face:
            self.conti_weights = torch.cat(weights, dim=0)+1 #0->1,1->2
    
    def get_patches_real(self):
        # [1,1,512,512]->[bs,1,11,11]
        patches = []
        t = np.random.randint(res,size=2)
        for i in range(aa):
            for j in range(aa):
                p = self.real_B[:,:,t[0]+i*W:t[0]+(i+1)*W,t[1]+j*W:t[1]+(j+1)*W]
                whitenum = torch.sum(p>=0.0)
                #if whitenum < 5 or whitenum > W*W-5:
                if whitenum < 1 or whitenum > W*W-1:  
                    continue
                patches.append(p)
        self.real_B_patches = torch.cat(patches, dim=0)