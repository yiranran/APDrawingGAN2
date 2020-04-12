from .base_model import BaseModel
from . import networks
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        # uncomment because default CycleGAN did not use dropout ( parser.set_defaults(no_dropout=True) )
        # parser = CycleGANModel.modify_commandline_options(parser, is_train=False)   
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')# no_lsgan=True, use_lsgan=False
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(auxiliary_root='auxiliaryeye2o')
        parser.set_defaults(use_local=True, hair_local=True, bg_local=True)
        parser.set_defaults(nose_ae=True, others_ae=True, compactmask=True, MOUTH_H=56)
        parser.set_defaults(soft_border=1)
        parser.add_argument('--nnG_hairc', type=int, default=6, help='nnG for hair classifier')
        parser.add_argument('--use_resnet', action='store_true', help='use resnet for generator')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]
        self.auxiliary_model_names = []
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

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        self.batch_size = len(self.image_paths)
        if self.opt.use_local:
            self.real_A_eyel = input['eyel_A'].to(self.device)
            self.real_A_eyer = input['eyer_A'].to(self.device)
            self.real_A_nose = input['nose_A'].to(self.device)
            self.real_A_mouth = input['mouth_A'].to(self.device)
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
            self.mask = input['mask'].to(self.device) # mask for non-eyes,nose,mouth
            self.mask2 = input['mask2'].to(self.device) # mask for non-bg
            self.real_A_bg = input['bg_A'].to(self.device)

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
            
            # HAIR, BG AND PARTCOMBINE
            outputs2 = self.netCLh(self.real_A_hair)
            onehot2 = self.getonehot(outputs2,3)

            fake_B_hair = self.netGLHair(self.real_A_hair,onehot2)
            fake_B_bg = self.netGLBG(self.real_A_bg)
            self.fake_B_hair = self.masked(fake_B_hair,self.mask*self.mask2)
            self.fake_B_bg = self.masked(fake_B_bg,self.inverse_mask(self.mask2))
            if not self.opt.compactmask:
                self.fake_B1 = self.partCombiner2_bg(fake_B_eyel,fake_B_eyer,fake_B_nose,fake_B_mouth,fake_B_hair,fake_B_bg,self.mask*self.mask2,self.inverse_mask(self.mask2),self.opt.comb_op)
            else:
                self.fake_B1 = self.partCombiner2_bg(fake_B_eyel,fake_B_eyer,fake_B_nose,fake_B_mouth,fake_B_hair,fake_B_bg,self.mask*self.mask2,self.inverse_mask(self.mask2),self.opt.comb_op,self.opt.region_enm,self.cmaskel,self.cmasker,self.cmask,self.cmaskmo)
            
            self.fake_B = self.netGCombine(torch.cat([self.fake_B0,self.fake_B1],1))
