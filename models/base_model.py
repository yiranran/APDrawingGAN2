import os
import torch
from collections import OrderedDict
from . import networks


class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.gpu_ids_p = opt.gpu_ids_p
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.device_p = torch.device('cuda:{}'.format(self.gpu_ids_p[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.auxiliary_dir = os.path.join(opt.checkpoints_dir, opt.auxiliary_root)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        if len(self.auxiliary_model_names) > 0:
            self.load_auxiliary_networks()
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
    
    def save_networks2(self, which_epoch):
        gen_name = os.path.join(self.save_dir, '%s_net_gen.pt' % (which_epoch))
        dis_name = os.path.join(self.save_dir, '%s_net_dis.pt' % (which_epoch))
        dict_gen = {}
        dict_dis = {}
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    state_dict = net.module.cpu().state_dict()
                    net.cuda(self.gpu_ids[0])
                else:
                    state_dict = net.cpu().state_dict()
                
                if name[0] == 'G':
                    dict_gen[name] = state_dict
                elif name[0] == 'D':
                    dict_dis[name] = state_dict
                else:
                    save_filename = '%s_net_%s.pth' % (which_epoch, name)
                    save_path = os.path.join(self.save_dir, save_filename)
                    torch.save(state_dict, save_path)
        if dict_gen:
            torch.save(dict_gen, gen_name)
        if dict_dis:
            torch.save(dict_dis, dis_name)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, which_epoch):
        gen_name = os.path.join(self.save_dir, '%s_net_gen.pt' % (which_epoch))
        if os.path.exists(gen_name):
            self.load_networks2(which_epoch)
            return
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
    
    def load_networks2(self, which_epoch):
        gen_name = os.path.join(self.save_dir, '%s_net_gen.pt' % (which_epoch))
        gen_state_dict = torch.load(gen_name, map_location=str(self.device))
        if self.isTrain and self.opt.model != 'apdrawing_style_nogan':
            dis_name = os.path.join(self.save_dir, '%s_net_dis.pt' % (which_epoch))
            dis_state_dict = torch.load(dis_name, map_location=str(self.device))
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                if name[0] == 'G':
                    print('loading the model %s from %s' % (name,gen_name))
                    state_dict = gen_state_dict[name]
                elif name[0] == 'D':
                    print('loading the model %s from %s' % (name,gen_name))
                    state_dict = dis_state_dict[name]
                
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
    
    # load auxiliary net models from the disk
    def load_auxiliary_networks(self):
        for name in self.auxiliary_model_names:
            if isinstance(name, str):
                if 'AE' in name and self.opt.ae_small:
                    load_filename = '%s_net_%s_small.pth' % ('latest', name)
                elif 'Regressor' in name:
                    load_filename = '%s_net_%s%d.pth' % ('latest', name, self.opt.regarch)
                else:
                    load_filename = '%s_net_%s.pth' % ('latest', name)
                load_path = os.path.join(self.auxiliary_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                if name in ['DT1', 'DT2', 'Line1', 'Line2', 'Continuity1', 'Continuity2', 'Regressor', 'Regressorhair', 'Regressorface']:
                    state_dict = torch.load(load_path, map_location=str(self.device_p))
                else:
                    state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # =============================================================================================================
    def inverse_mask(self, mask):
        return torch.ones(mask.shape).to(self.device)-mask
    
    def masked(self, A,mask):
        return (A/2+0.5)*mask*2-1
    
    def add_with_mask(self, A,B,mask):
        return ((A/2+0.5)*mask+(B/2+0.5)*(torch.ones(mask.shape).to(self.device)-mask))*2-1
    
    def addone_with_mask(self, A,mask):
        return ((A/2+0.5)*mask+(torch.ones(mask.shape).to(self.device)-mask))*2-1

    def partCombiner(self, eyel, eyer, nose, mouth, average_pos=False, comb_op = 1, region_enm = 0, cmaskel = None, cmasker = None, cmaskno = None, cmaskmo = None):
        '''
        x         y
        100.571   123.429
        155.429   123.429
        128.000   155.886
        103.314   185.417
        152.686   185.417
        this is the mean locaiton of 5 landmarks (for 256x256)
        Pad2d Left,Right,Top,Down
        '''
        if comb_op == 0:
            # use max pooling, pad black for eyes etc
            padvalue = -1
            if region_enm in [1,2]:
                eyel = eyel * cmaskel
                eyer = eyer * cmasker
                nose = nose * cmaskno
                mouth = mouth * cmaskmo
        else:
            # use min pooling, pad white for eyes etc
            padvalue = 1
            if region_enm in [1,2]:
                eyel = self.addone_with_mask(eyel, cmaskel)
                eyer = self.addone_with_mask(eyer, cmasker)
                nose = self.addone_with_mask(nose, cmaskno)
                mouth = self.addone_with_mask(mouth, cmaskmo)
        if region_enm in [0,1]: # need to pad
            IMAGE_SIZE = self.opt.fineSize
            ratio = IMAGE_SIZE / 256
            EYE_W = self.opt.EYE_W * ratio
            EYE_H = self.opt.EYE_H * ratio
            NOSE_W = self.opt.NOSE_W * ratio
            NOSE_H = self.opt.NOSE_H * ratio
            MOUTH_W = self.opt.MOUTH_W * ratio
            MOUTH_H = self.opt.MOUTH_H * ratio
            bs,nc,_,_ = eyel.shape
            eyel_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            eyer_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            nose_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            mouth_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            for i in range(bs):
                if not average_pos:
                    center = self.center[i]#x,y
                else:# if average_pos = True
                    center = torch.tensor([[101,123-4],[155,123-4],[128,156-NOSE_H/2+16],[128,185]])
                eyel_p[i] = torch.nn.ConstantPad2d((int(center[0,0] - EYE_W / 2 - 1), int(IMAGE_SIZE - (center[0,0]+EYE_W/2-1)), int(center[0,1] - EYE_H / 2 - 1),int(IMAGE_SIZE - (center[0,1]+EYE_H/2 - 1))),-1)(eyel[i])
                eyer_p[i] = torch.nn.ConstantPad2d((int(center[1,0] - EYE_W / 2 - 1), int(IMAGE_SIZE - (center[1,0]+EYE_W/2-1)), int(center[1,1] - EYE_H / 2 - 1), int(IMAGE_SIZE - (center[1,1]+EYE_H/2 - 1))),-1)(eyer[i])
                nose_p[i] = torch.nn.ConstantPad2d((int(center[2,0] - NOSE_W / 2 - 1), int(IMAGE_SIZE - (center[2,0]+NOSE_W/2-1)), int(center[2,1] - NOSE_H / 2 - 1), int(IMAGE_SIZE - (center[2,1]+NOSE_H/2 - 1))),-1)(nose[i])
                mouth_p[i] = torch.nn.ConstantPad2d((int(center[3,0] - MOUTH_W / 2 - 1), int(IMAGE_SIZE - (center[3,0]+MOUTH_W/2-1)), int(center[3,1] - MOUTH_H / 2 - 1), int(IMAGE_SIZE - (center[3,1]+MOUTH_H/2 - 1))),-1)(mouth[i])
        elif region_enm in [2]:
            eyel_p = eyel
            eyer_p = eyer
            nose_p = nose
            mouth_p = mouth
        if comb_op == 0:
            # use max pooling
            eyes = torch.max(eyel_p, eyer_p)
            eye_nose = torch.max(eyes, nose_p)
            result = torch.max(eye_nose, mouth_p)
        else:
            # use min pooling
            eyes = torch.min(eyel_p, eyer_p)
            eye_nose = torch.min(eyes, nose_p)
            result = torch.min(eye_nose, mouth_p)
        return result
    
    def partCombiner2(self, eyel, eyer, nose, mouth, hair, mask, comb_op = 1, region_enm = 0, cmaskel = None, cmasker = None, cmaskno = None, cmaskmo = None):
        if comb_op == 0:
            # use max pooling, pad black for eyes etc
            padvalue = -1
            hair = self.masked(hair, mask)
            if region_enm in [1,2]:
                eyel = eyel * cmaskel
                eyer = eyer * cmasker
                nose = nose * cmaskno
                mouth = mouth * cmaskmo
        else:
            # use min pooling, pad white for eyes etc
            padvalue = 1
            hair = self.addone_with_mask(hair, mask)
            if region_enm in [1,2]:
                eyel = self.addone_with_mask(eyel, cmaskel)
                eyer = self.addone_with_mask(eyer, cmasker)
                nose = self.addone_with_mask(nose, cmaskno)
                mouth = self.addone_with_mask(mouth, cmaskmo)
        if region_enm in [0,1]: # need to pad
            IMAGE_SIZE = self.opt.fineSize
            ratio = IMAGE_SIZE / 256
            EYE_W = self.opt.EYE_W * ratio
            EYE_H = self.opt.EYE_H * ratio
            NOSE_W = self.opt.NOSE_W * ratio
            NOSE_H = self.opt.NOSE_H * ratio
            MOUTH_W = self.opt.MOUTH_W * ratio
            MOUTH_H = self.opt.MOUTH_H * ratio
            bs,nc,_,_ = eyel.shape
            eyel_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            eyer_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            nose_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            mouth_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            for i in range(bs):
                center = self.center[i]#x,y
                eyel_p[i] = torch.nn.ConstantPad2d((center[0,0] - EYE_W / 2, IMAGE_SIZE - (center[0,0]+EYE_W/2), center[0,1] - EYE_H / 2, IMAGE_SIZE - (center[0,1]+EYE_H/2)),padvalue)(eyel[i])
                eyer_p[i] = torch.nn.ConstantPad2d((center[1,0] - EYE_W / 2, IMAGE_SIZE - (center[1,0]+EYE_W/2), center[1,1] - EYE_H / 2, IMAGE_SIZE - (center[1,1]+EYE_H/2)),padvalue)(eyer[i])
                nose_p[i] = torch.nn.ConstantPad2d((center[2,0] - NOSE_W / 2, IMAGE_SIZE - (center[2,0]+NOSE_W/2), center[2,1] - NOSE_H / 2, IMAGE_SIZE - (center[2,1]+NOSE_H/2)),padvalue)(nose[i])
                mouth_p[i] = torch.nn.ConstantPad2d((center[3,0] - MOUTH_W / 2, IMAGE_SIZE - (center[3,0]+MOUTH_W/2), center[3,1] - MOUTH_H / 2, IMAGE_SIZE - (center[3,1]+MOUTH_H/2)),padvalue)(mouth[i])
        elif region_enm in [2]:
            eyel_p = eyel
            eyer_p = eyer
            nose_p = nose
            mouth_p = mouth
        if comb_op == 0:
            # use max pooling
            eyes = torch.max(eyel_p, eyer_p)
            eye_nose = torch.max(eyes, nose_p)
            eye_nose_mouth = torch.max(eye_nose, mouth_p)
            result = torch.max(hair,eye_nose_mouth)
        else:
            # use min pooling
            eyes = torch.min(eyel_p, eyer_p)
            eye_nose = torch.min(eyes, nose_p)
            eye_nose_mouth = torch.min(eye_nose, mouth_p)
            result = torch.min(hair,eye_nose_mouth)
        return result
    
    def partCombiner2_bg(self, eyel, eyer, nose, mouth, hair, bg, maskh, maskb, comb_op = 1, region_enm = 0, cmaskel = None, cmasker = None, cmaskno = None, cmaskmo = None):
        if comb_op == 0:
            # use max pooling, pad black for eyes etc
            padvalue = -1
            hair = self.masked(hair, maskh)
            bg = self.masked(bg, maskb)
            if region_enm in [1,2]:
                eyel = eyel * cmaskel
                eyer = eyer * cmasker
                nose = nose * cmaskno
                mouth = mouth * cmaskmo
        else:
            # use min pooling, pad white for eyes etc
            padvalue = 1
            hair = self.addone_with_mask(hair, maskh)
            bg = self.addone_with_mask(bg, maskb)
            if region_enm in [1,2]:
                eyel = self.addone_with_mask(eyel, cmaskel)
                eyer = self.addone_with_mask(eyer, cmasker)
                nose = self.addone_with_mask(nose, cmaskno)
                mouth = self.addone_with_mask(mouth, cmaskmo)
        if region_enm in [0,1]: # need to pad to full size
            IMAGE_SIZE = self.opt.fineSize
            ratio = IMAGE_SIZE / 256
            EYE_W = self.opt.EYE_W * ratio
            EYE_H = self.opt.EYE_H * ratio
            NOSE_W = self.opt.NOSE_W * ratio
            NOSE_H = self.opt.NOSE_H * ratio
            MOUTH_W = self.opt.MOUTH_W * ratio
            MOUTH_H = self.opt.MOUTH_H * ratio
            bs,nc,_,_ = eyel.shape
            eyel_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            eyer_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            nose_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            mouth_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
            for i in range(bs):
                center = self.center[i]#x,y
                eyel_p[i] = torch.nn.ConstantPad2d((int(center[0,0] - EYE_W / 2), int(IMAGE_SIZE - (center[0,0]+EYE_W/2)), int(center[0,1] - EYE_H / 2), int(IMAGE_SIZE - (center[0,1]+EYE_H/2))),padvalue)(eyel[i])
                eyer_p[i] = torch.nn.ConstantPad2d((int(center[1,0] - EYE_W / 2), int(IMAGE_SIZE - (center[1,0]+EYE_W/2)), int(center[1,1] - EYE_H / 2), int(IMAGE_SIZE - (center[1,1]+EYE_H/2))), padvalue)(eyer[i])
                nose_p[i] = torch.nn.ConstantPad2d((int(center[2,0] - NOSE_W / 2), int(IMAGE_SIZE - (center[2,0]+NOSE_W/2)), int(center[2,1] - NOSE_H / 2), int(IMAGE_SIZE - (center[2,1]+NOSE_H/2))),padvalue)(nose[i])
                mouth_p[i] = torch.nn.ConstantPad2d((int(center[3,0] - MOUTH_W / 2), int(IMAGE_SIZE - (center[3,0]+MOUTH_W/2)), int(center[3,1] - MOUTH_H / 2), int(IMAGE_SIZE - (center[3,1]+MOUTH_H/2))),padvalue)(mouth[i])
        
        elif region_enm in [2]:
            eyel_p = eyel
            eyer_p = eyer
            nose_p = nose
            mouth_p = mouth
        if comb_op == 0:
            eyes = torch.max(eyel_p, eyer_p)
            eye_nose = torch.max(eyes, nose_p)
            eye_nose_mouth = torch.max(eye_nose, mouth_p)
            eye_nose_mouth_hair = torch.max(hair,eye_nose_mouth)
            result = torch.max(bg,eye_nose_mouth_hair)
        else:
            eyes = torch.min(eyel_p, eyer_p)
            eye_nose = torch.min(eyes, nose_p)
            eye_nose_mouth = torch.min(eye_nose, mouth_p)
            eye_nose_mouth_hair = torch.min(hair,eye_nose_mouth)
            result = torch.min(bg,eye_nose_mouth_hair)
        return result
    
    def partCombiner3(self, face, hair, maskf, maskh, comb_op = 1):
        if comb_op == 0:
            # use max pooling, pad black etc
            padvalue = -1
            face = self.masked(face, maskf)
            hair = self.masked(hair, maskh)
        else:
            # use min pooling, pad white etc
            padvalue = 1
            face = self.addone_with_mask(face, maskf)
            hair = self.addone_with_mask(hair, maskh)
        if comb_op == 0:
            result = torch.max(face,hair)
        else:
            result = torch.min(face,hair)
        return result


    def tocv2(ts):
        img = (ts.numpy()/2+0.5)*255
        img = img.astype('uint8')
        img = np.transpose(img,(1,2,0))
        img = img[:,:,::-1]#rgb->bgr
        return img
    
    def totor(img):
        img = img[:,:,::-1]
        tor = transforms.ToTensor()(img)
        tor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tor)
        return tor
    
    
    def ContinuityForTest(self, real = 0):
        # Patch-based
        self.get_patches()
        self.outputs = self.netRegressor(self.fake_B_patches)
        line_continuity = torch.mean(self.outputs)
        opt = self.opt
        file_name = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'continuity.txt')
        message = '%s %.04f' % (self.image_paths[0], line_continuity)
        with open(file_name, 'a+') as c_file:
            c_file.write(message)
            c_file.write('\n')
        if real == 1:
            self.get_patches_real()
            self.outputs2 = self.netRegressor(self.real_B_patches)
            line_continuity2 = torch.mean(self.outputs2)
            file_name = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'continuity-r.txt')
            message = '%s %.04f' % (self.image_paths[0], line_continuity2)
            with open(file_name, 'a+') as c_file:
                c_file.write(message)
                c_file.write('\n')
    
    def getLocalParts(self,fakeAB):
        bs,nc,_,_ = fakeAB.shape #dtype torch.float32
        ncr = int(nc / self.opt.output_nc)
        if self.opt.region_enm in [0,1]:
            ratio = self.opt.fineSize / 256
            EYE_H = self.opt.EYE_H * ratio
            EYE_W = self.opt.EYE_W * ratio
            NOSE_H = self.opt.NOSE_H * ratio
            NOSE_W = self.opt.NOSE_W * ratio
            MOUTH_H = self.opt.MOUTH_H * ratio
            MOUTH_W = self.opt.MOUTH_W * ratio
            eyel = torch.ones((bs,nc,int(EYE_H),int(EYE_W))).to(self.device)
            eyer = torch.ones((bs,nc,int(EYE_H),int(EYE_W))).to(self.device)
            nose = torch.ones((bs,nc,int(NOSE_H),int(NOSE_W))).to(self.device)
            mouth = torch.ones((bs,nc,int(MOUTH_H),int(MOUTH_W))).to(self.device)
            for i in range(bs):
                center = self.center[i]
                eyel[i] = fakeAB[i,:,int(center[0,1]-EYE_H/2):int(center[0,1]+EYE_H/2),int(center[0,0]-EYE_W/2):int(center[0,0]+EYE_W/2)]
                eyer[i] = fakeAB[i,:,int(center[1,1]-EYE_H/2):int(center[1,1]+EYE_H/2),int(center[1,0]-EYE_W/2):int(center[1,0]+EYE_W/2)]
                nose[i] = fakeAB[i,:,int(center[2,1]-NOSE_H/2):int(center[2,1]+NOSE_H/2),int(center[2,0]-NOSE_W/2):int(center[2,0]+NOSE_W/2)]
                mouth[i] = fakeAB[i,:,int(center[3,1]-MOUTH_H/2):int(center[3,1]+MOUTH_H/2),int(center[3,0]-MOUTH_W/2):int(center[3,0]+MOUTH_W/2)]
        elif self.opt.region_enm in [2]:
            eyel = (fakeAB/2+0.5) * self.cmaskel.repeat(1,ncr,1,1) * 2 - 1
            eyer = (fakeAB/2+0.5) * self.cmasker.repeat(1,ncr,1,1) * 2 - 1
            nose = (fakeAB/2+0.5) * self.cmask.repeat(1,ncr,1,1) * 2 - 1
            mouth = (fakeAB/2+0.5) * self.cmaskmo.repeat(1,ncr,1,1) * 2 - 1
        hair = (fakeAB/2+0.5) * self.mask.repeat(1,ncr,1,1) * self.mask2.repeat(1,ncr,1,1) * 2 - 1
        bg = (fakeAB/2+0.5) * (torch.ones(fakeAB.shape).to(self.device)-self.mask2.repeat(1,ncr,1,1)) * 2 - 1
        return eyel, eyer, nose, mouth, hair, bg
    
    def getaddw(self,local_name):
        addw = 1
        if local_name in ['DLEyel','DLEyer','eyel','eyer','DLFace','face']:
            addw = self.opt.addw_eye
        elif local_name in ['DLNose', 'nose']:
            addw = self.opt.addw_nose
        elif local_name in ['DLMouth', 'mouth']:
            addw = self.opt.addw_mouth
        elif local_name in ['DLHair', 'hair']:
            addw = self.opt.addw_hair
        elif local_name in ['DLBG', 'bg']:
            addw = self.opt.addw_bg
        return addw
