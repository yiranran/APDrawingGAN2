import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
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
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], nnG=9, multiple=2, latent_dim=1024, ae_h=96, ae_w=96, extra_channel=2, nres=1):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'autoencoder':
        net = AutoEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'autoencoderfc':
        net = AutoEncoderWithFC(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
        multiple=multiple, latent_dim=latent_dim, h=ae_h, w=ae_w)
    elif netG == 'autoencoderfc2':
        net = AutoEncoderWithFC2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
        multiple=multiple, latent_dim=latent_dim, h=ae_h, w=ae_w)
    elif netG == 'vae':
        net = VAE(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
        multiple=multiple, latent_dim=latent_dim, h=ae_h, w=ae_w)
    elif netG == 'classifier':
        net = Classifier(input_nc, output_nc, ngf, num_downs=nnG, norm_layer=norm_layer, use_dropout=use_dropout, h=ae_h, w=ae_w)
    elif netG == 'regressor':
        net = Regressor(input_nc, ngf, norm_layer=norm_layer, arch=nnG)
    elif netG == 'resnet_9blocks':#default for cyclegan
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_nblocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=nnG)
    elif netG == 'resnet_style2_9blocks':
        net = ResnetStyle2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, model0_res=0, extra_channel=extra_channel)
    elif netG == 'resnet_style2_6blocks':
        net = ResnetStyle2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, model0_res=0, extra_channel=extra_channel)
    elif netG == 'resnet_style2_nblocks':
        net = ResnetStyle2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=nnG, model0_res=0, extra_channel=extra_channel)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':#default for pix2pix
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_ndown':
        net = UnetGenerator(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unetres_ndown':
        net = UnetResGenerator(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout, nres=nres)
    elif netG == 'partunet':
        net = PartUnet(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'partunet2':
        net = PartUnet2(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'partunetres':
        net = PartUnetRes(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout,nres=nres)
    elif netG == 'partunet2res':
        net = PartUnet2Res(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout,nres=nres)
    elif netG == 'partunet2style':
        net = PartUnet2Style(input_nc, output_nc, nnG, ngf, extra_channel=extra_channel, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'partunet2resstyle':
        net = PartUnet2ResStyle(input_nc, output_nc, nnG, ngf, extra_channel=extra_channel, norm_layer=norm_layer, use_dropout=use_dropout,nres=nres)
    elif netG == 'combiner':
         net = Combiner(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=2)
    elif netG == 'combiner2':
        net = Combiner2(input_nc, output_nc, nnG, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:#no_lsgan
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class AutoEncoderMNIST(nn.Module):
    def __init__(self):
        super(AutoEncoderMNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect'):
        super(AutoEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.LeakyReLU(0.2),
                      nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2)]
        self.encoder = nn.Sequential(*model)

        model2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model2 += [nn.ReLU(),
                       nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / 2))]
        model2 += [nn.ReLU()]
        model2 += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model2 += [nn.Tanh()]
        self.decoder = nn.Sequential(*model2)

    def forward(self, x):
        ax = self.encoder(x) # b, 512, 6, 6
        y = self.decoder(ax)
        return y, ax

class AutoEncoderWithFC(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, multiple=2,latent_dim=1024,  h=96, w=96):
        super(AutoEncoderWithFC, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        n_downsampling = 3
        #multiple = 2
        for i in range(n_downsampling):
            mult = multiple**i
            model += [nn.LeakyReLU(0.2),
                      nn.Conv2d(int(ngf * mult), int(ngf * mult * multiple), kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult * multiple))]
        self.encoder = nn.Sequential(*model)
        self.fc1 = nn.Linear(int(ngf*(multiple**n_downsampling)*h/16*w/16),latent_dim)
        self.relu = nn.ReLU(latent_dim)
        self.fc2 = nn.Linear(latent_dim,int(ngf*(multiple**n_downsampling)*h/16*w/16))
        self.rh = int(h/16)
        self.rw = int(w/16)
        model2 = []
        for i in range(n_downsampling):
            mult = multiple**(n_downsampling - i)
            model2 += [nn.ReLU(),
                       nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / multiple),
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / multiple))]
        model2 += [nn.ReLU()]
        model2 += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model2 += [nn.Tanh()]
        self.decoder = nn.Sequential(*model2)

    def forward(self, x):
        ax = self.encoder(x) # b, 512, 6, 6
        ax = ax.view(ax.size(0), -1) # view -- reshape
        ax = self.relu(self.fc1(ax))
        ax = self.fc2(ax)
        ax = ax.view(ax.size(0),-1,self.rh,self.rw)
        y = self.decoder(ax)
        return y, ax
    
class AutoEncoderWithFC2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, multiple=2,latent_dim=1024,  h=96, w=96):
        super(AutoEncoderWithFC2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        n_downsampling = 2
        #multiple = 2
        for i in range(n_downsampling):
            mult = multiple**i
            model += [nn.LeakyReLU(0.2),
                      nn.Conv2d(int(ngf * mult), int(ngf * mult * multiple), kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult * multiple))]
        self.encoder = nn.Sequential(*model)
        self.fc1 = nn.Linear(int(ngf*(multiple**n_downsampling)*h/8*w/8),latent_dim)
        self.relu = nn.ReLU(latent_dim)
        self.fc2 = nn.Linear(latent_dim,int(ngf*(multiple**n_downsampling)*h/8*w/8))
        self.rh = h/8
        self.rw = w/8
        model2 = []
        for i in range(n_downsampling):
            mult = multiple**(n_downsampling - i)
            model2 += [nn.ReLU(),
                       nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / multiple),
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / multiple))]
        model2 += [nn.ReLU()]
        model2 += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model2 += [nn.Tanh()]
        self.decoder = nn.Sequential(*model2)

    def forward(self, x):
        ax = self.encoder(x) # b, 256, 12, 12
        ax = ax.view(ax.size(0), -1) # view -- reshape
        ax = self.relu(self.fc1(ax))
        ax = self.fc2(ax)
        ax = ax.view(ax.size(0),-1,self.rh,self.rw)
        y = self.decoder(ax)
        return y, ax

class VAE(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, multiple=2,latent_dim=1024,  h=96, w=96):
        super(VAE, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = multiple**i
            model += [nn.LeakyReLU(0.2),
                      nn.Conv2d(int(ngf * mult), int(ngf * mult * multiple), kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult * multiple))]
        self.encoder_cnn = nn.Sequential(*model)

        self.c_dim = int(ngf*(multiple**n_downsampling)*h/16*w/16)
        self.rh = h/16
        self.rw = w/16
        self.fc1 = nn.Linear(self.c_dim,latent_dim)
        self.fc2 = nn.Linear(self.c_dim,latent_dim)
        self.fc3 = nn.Linear(latent_dim,self.c_dim)
        self.relu = nn.ReLU()

        model2 = []
        for i in range(n_downsampling):
            mult = multiple**(n_downsampling - i)
            model2 += [nn.ReLU(),
                       nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / multiple),
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / multiple))]
        model2 += [nn.ReLU()]
        model2 += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model2 += [nn.Tanh()]#[-1,1]
        self.decoder_cnn = nn.Sequential(*model2)
    
    def encode(self, x):
        h1 = self.encoder_cnn(x)
        r1 = h1.view(h1.size(0), -1)
        return self.fc1(r1), self.fc2(r1)
    
    def reparameterize(self, mu, logvar):# not deterministic for test mode
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)# torch.rand_like returns a tensor with the same size as input,
                                   # that is filled with random numbers from a normal distribution N(0,1)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        h4 = self.relu(self.fc3(z))
        r3 = h4.view(h4.size(0),-1,self.rh,self.rw)
        return self.decoder_cnn(r3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconx = self.decode(z)
        return reconx, mu, logvar

class Classifier(nn.Module):
    def __init__(self, input_nc, classes, ngf=64, num_downs=3, norm_layer=nn.BatchNorm2d, use_dropout=False,
        h=96, w=96):
        super(Classifier, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        multiple = 2
        for i in range(num_downs):
            mult = multiple**i
            model += [nn.LeakyReLU(0.2),
                      nn.Conv2d(int(ngf * mult), int(ngf * mult * multiple), kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult * multiple))]
        self.encoder = nn.Sequential(*model)
        strides = 2**(num_downs+1)
        self.fc1 = nn.Linear(int(ngf*h*w/(strides*2)), classes)

    def forward(self, x):
        ax = self.encoder(x) # b, 512, 6, 6
        ax = ax.view(ax.size(0), -1) # view -- reshape
        return self.fc1(ax)

class Regressor(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, arch=1):
        super(Regressor, self).__init__()
        # if use BatchNorm2d, 
        # no need to use bias as BatchNorm2d has affine parameters

        self.arch = arch
        
        if arch == 1:
            use_bias = True
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),#11->5
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, 1, kernel_size=5, stride=1, padding=0, bias=use_bias),#5->1
            ]
        elif arch == 2:
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=0, bias=use_bias),#11->9
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=0, bias=use_bias),#9->7
                norm_layer(ngf*2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=1, padding=0, bias=use_bias),#7->5
                norm_layer(ngf*4),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*4, 1, kernel_size=5, stride=1, padding=0, bias=use_bias),#5->1
            ]
        elif arch == 3:
            use_bias = True
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, 1, kernel_size=11, stride=1, padding=0, bias=use_bias),#11->1
            ]
        elif arch == 4:
            use_bias = True
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*4, 1, kernel_size=11, stride=1, padding=0, bias=use_bias),#11->1
            ]
        elif arch == 5:
            use_bias = True
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
            ]
            fc = [
                nn.Linear(ngf*4*11*11, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 1),
            ]
            self.fc = nn.Sequential(*fc)

        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        if self.arch <= 4:
            return self.model(x)
        else:
            x = self.model(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetStyle2Generator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', extra_channel=3, model0_res=0):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetStyle2Generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model0 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        mult = 2 ** n_downsampling
        for i in range(model0_res):       # add ResNet blocks
            model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model = []
        model += [nn.Conv2d(ngf * mult + extra_channel, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf * mult),
                      nn.ReLU(True)]

        for i in range(n_blocks-model0_res):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model0 = nn.Sequential(*model0)
        self.model = nn.Sequential(*model)
        print(list(self.modules()))

    def forward(self, input1, input2): # input2 [bs,c]
        """Standard forward"""
        f1 = self.model0(input1)
        [bs,c,h,w] = f1.shape
        input2 = input2.repeat(h,w,1,1).permute([2,3,0,1])
        y1 = torch.cat([f1, input2], 1)
        return self.model(y1)

class Combiner(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Combiner, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_blocks):
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Combiner2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Combiner2, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetResGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, nres=1):
        super(UnetResGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionResBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, nres=nres)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class PartUnet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(PartUnet, self).__init__()

        # construct unet structure
        # 3 downs 
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class PartUnetRes(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, nres=1):
        super(PartUnetRes, self).__init__()

        # construct unet structure
        # 3 downs 
        unet_block = UnetSkipConnectionResBlock(ngf * 2, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, nres=nres)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class PartUnet2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(PartUnet2, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 3):
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class PartUnet2Res(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, nres=1):
        super(PartUnet2Res, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionResBlock(ngf * 2, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, nres=nres)
        for i in range(num_downs - 3):
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class PartUnet2Style(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, extra_channel=2,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(PartUnet2Style, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionStyleBlock(ngf * 2, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, extra_channel=extra_channel)
        for i in range(num_downs - 3):
            unet_block = UnetSkipConnectionStyleBlock(ngf * 2, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, extra_channel=extra_channel)
        unet_block = UnetSkipConnectionStyleBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, extra_channel=extra_channel)
        unet_block = UnetSkipConnectionStyleBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, extra_channel=extra_channel)

        self.model = unet_block

    def forward(self, input, cate):
        return self.model(input, cate)

class PartUnet2ResStyle(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, extra_channel=2,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, nres=1):
        super(PartUnet2ResStyle, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionResStyleBlock(ngf * 2, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, extra_channel=extra_channel, nres=nres)
        for i in range(num_downs - 3):
            unet_block = UnetSkipConnectionStyleBlock(ngf * 2, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, extra_channel=extra_channel)
        unet_block = UnetSkipConnectionStyleBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, extra_channel=extra_channel)
        unet_block = UnetSkipConnectionStyleBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, extra_channel=extra_channel)

        self.model = unet_block

    def forward(self, input, cate):
        return self.model(input, cate)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionResBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, nres=1):
        super(UnetSkipConnectionResBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downrelu]
            up = [upconv, upnorm]
            model = down
            # resblock: conv norm relu conv norm +
            for i in range(nres):
                model += [ResnetBlock(inner_nc, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            model += up
            #model = down + [submodule] + up
            print('UnetSkipConnectionResBlock','nres',nres,'inner_nc',inner_nc)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionStyleBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 extra_channel=2, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionStyleBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.extra_channel = extra_channel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc+extra_channel, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

        self.downmodel = nn.Sequential(*down)
        self.upmodel = nn.Sequential(*up)
        self.submodule = submodule

    def forward(self, x, cate):# cate [bs,c]
        if self.innermost:
            y1 = self.downmodel(x)
            [bs,c,h,w] = y1.shape
            map = cate.repeat(h,w,1,1).permute([2,3,0,1])
            y2 = torch.cat([y1,map], 1)
            y3 = self.upmodel(y2)
            return torch.cat([x, y3], 1)
        else:
            y1 = self.downmodel(x)
            y2 = self.submodule(y1,cate)
            y3 = self.upmodel(y2)
            if self.outermost:
                return y3
            else:
                return torch.cat([x, y3], 1)

class UnetSkipConnectionResStyleBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 extra_channel=2, norm_layer=nn.BatchNorm2d, use_dropout=False, nres=1):
        super(UnetSkipConnectionResStyleBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.extra_channel = extra_channel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downrelu]
            up = [nn.Conv2d(inner_nc+extra_channel, inner_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(inner_nc),
                      nn.ReLU(True)]
            for i in range(nres):
                up += [ResnetBlock(inner_nc, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            up += [ upconv, upnorm]
            model = down + up
            print('UnetSkipConnectionResStyleBlock','nres',nres,'inner_nc',inner_nc)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

        self.downmodel = nn.Sequential(*down)
        self.upmodel = nn.Sequential(*up)
        self.submodule = submodule

    def forward(self, x, cate):# cate [bs,c]
        # concate in the innermost block
        if self.innermost:
            y1 = self.downmodel(x)
            [bs,c,h,w] = y1.shape
            map = cate.repeat(h,w,1,1).permute([2,3,0,1])
            y2 = torch.cat([y1,map], 1)
            y3 = self.upmodel(y2)
            return torch.cat([x, y3], 1)
        else:
            y1 = self.downmodel(x)
            y2 = self.submodule(y1,cate)
            y3 = self.upmodel(y2)
            if self.outermost:
                return y3
            else:
                return torch.cat([x, y3], 1)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:#no_lsgan, use sigmoid before calculating bceloss(binary cross entropy)
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
