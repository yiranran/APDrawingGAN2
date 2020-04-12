import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import csv
import torch
import torchvision.transforms as transforms

def getfeats(featpath):
	trans_points = np.empty([5,2],dtype=np.int64) 
	with open(featpath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for ind,row in enumerate(reader):
			trans_points[ind,:] = row
	return trans_points

def getSoft(size,xb,yb,boundwidth=5.0):
    xarray = np.tile(np.arange(0,size[1]),(size[0],1))
    yarray = np.tile(np.arange(0,size[0]),(size[1],1)).transpose()
    cxdists = []
    cydists = []
    for i in range(len(xb)):
        xba = np.tile(xb[i],(size[1],1)).transpose()
        yba = np.tile(yb[i],(size[0],1))
        cxdists.append(np.abs(xarray-xba))
        cydists.append(np.abs(yarray-yba))
    xdist = np.minimum.reduce(cxdists)
    ydist = np.minimum.reduce(cydists)
    manhdist = np.minimum.reduce([xdist,ydist])
    im = (manhdist+1) / (boundwidth+1) * 1.0
    im[im>=1.0] = 1.0
    return im

class SingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)
        imglist = 'datasets/apdrawing_list/%s/%s.txt' % (opt.phase, opt.dataroot)
        if os.path.exists(imglist):
            lines = open(imglist, 'r').read().splitlines()
            self.A_paths = sorted(lines)
        else:
            self.A_paths = make_dataset(self.dir_A)
            self.A_paths = sorted(self.A_paths)
        self.transform = get_transform(opt) # this function uses NO_FLIP; aligned dataset do not use this, aligned dataset manually transform

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        item = {'A': A, 'A_paths': A_path}

        if self.opt.use_local:
            regions = ['eyel','eyer','nose','mouth']
            basen = os.path.basename(A_path)[:-4]+'.txt'
            featdir = self.opt.lm_dir
            featpath = os.path.join(featdir,basen)
            feats = getfeats(featpath)
            mouth_x = int((feats[3,0]+feats[4,0])/2.0)
            mouth_y = int((feats[3,1]+feats[4,1])/2.0)
            ratio = self.opt.fineSize / 256
            EYE_H = self.opt.EYE_H * ratio
            EYE_W = self.opt.EYE_W * ratio
            NOSE_H = self.opt.NOSE_H * ratio
            NOSE_W = self.opt.NOSE_W * ratio
            MOUTH_H = self.opt.MOUTH_H * ratio
            MOUTH_W = self.opt.MOUTH_W * ratio
            center = torch.IntTensor([[feats[0,0],feats[0,1]-4*ratio],[feats[1,0],feats[1,1]-4*ratio],[feats[2,0],feats[2,1]-NOSE_H/2+16*ratio],[mouth_x,mouth_y]])
            item['center'] = center
            rhs = [int(EYE_H),int(EYE_H),int(NOSE_H),int(MOUTH_H)]
            rws = [int(EYE_W),int(EYE_W),int(NOSE_W),int(MOUTH_W)]
            if self.opt.soft_border:
                soft_border_mask4 = []
                for i in range(4):
                    xb = [np.zeros(rhs[i]),np.ones(rhs[i])*(rws[i]-1)]
                    yb = [np.zeros(rws[i]),np.ones(rws[i])*(rhs[i]-1)]
                    soft_border_mask = getSoft([rhs[i],rws[i]],xb,yb)
                    soft_border_mask4.append(torch.Tensor(soft_border_mask).unsqueeze(0))
                    item['soft_'+regions[i]+'_mask'] = soft_border_mask4[i]
            for i in range(4):
                item[regions[i]+'_A'] = A[:,center[i,1]-rhs[i]/2:center[i,1]+rhs[i]/2,center[i,0]-rws[i]/2:center[i,0]+rws[i]/2]
                if self.opt.soft_border:
                    item[regions[i]+'_A'] = item[regions[i]+'_A'] * soft_border_mask4[i].repeat(int(input_nc/output_nc),1,1)
            if self.opt.compactmask:
                cmasks0 = []
                cmasks = []
                for i in range(4):
                    cmaskpath = os.path.join(self.opt.cmask_dir,regions[i],basen[:-4]+'.png')
                    im_cmask = Image.open(cmaskpath)
                    cmask0 = transforms.ToTensor()(im_cmask)
                    if output_nc == 1 and cmask0.shape[0] == 3:
                        tmp = cmask0[0, ...] * 0.299 + cmask0[1, ...] * 0.587 + cmask0[2, ...] * 0.114
                        cmask0 = tmp.unsqueeze(0)
                    cmask0 = (cmask0 >= 0.5).float()
                    cmasks0.append(cmask0)
                    cmask = cmask0.clone()
                    cmask = cmask[:,center[i,1]-rhs[i]/2:center[i,1]+rhs[i]/2,center[i,0]-rws[i]/2:center[i,0]+rws[i]/2]
                    cmasks.append(cmask)
                item['cmaskel'] = cmasks[0]
                item['cmasker'] = cmasks[1]
                item['cmask'] = cmasks[2]
                item['cmaskmo'] = cmasks[3]
            if self.opt.hair_local:
                output_nc = self.opt.output_nc
                mask = torch.ones([output_nc,A.shape[1],A.shape[2]])
                for i in range(4):
                    mask[:,center[i,1]-rhs[i]/2:center[i,1]+rhs[i]/2,center[i,0]-rws[i]/2:center[i,0]+rws[i]/2] = 0
                if self.opt.soft_border:
                    imgsize = self.opt.fineSize
                    maskn = mask[0].numpy()
                    masks = [np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize])]
                    masks[0][1:] = maskn[:-1]
                    masks[1][:-1] = maskn[1:]
                    masks[2][:,1:] = maskn[:,:-1]
                    masks[3][:,:-1] = maskn[:,1:]
                    masks2 = [maskn-e for e in masks]
                    bound = np.minimum.reduce(masks2)
                    bound = -bound
                    xb = []
                    yb = []
                    for i in range(4):
                        xbi = [center[i,0]-rws[i]/2, center[i,0]+rws[i]/2-1]
                        ybi = [center[i,1]-rhs[i]/2, center[i,1]+rhs[i]/2-1]
                        for j in range(2):
                            maskx = bound[:,xbi[j]]
                            masky = bound[ybi[j],:]
                            tmp_a = torch.from_numpy(maskx)*xbi[j].double()
                            tmp_b = torch.from_numpy(1-maskx)
                            xb += [tmp_b*10000 + tmp_a]

                            tmp_a = torch.from_numpy(masky)*ybi[j].double()
                            tmp_b = torch.from_numpy(1-masky)
                            yb += [tmp_b*10000 + tmp_a]
                    soft = 1-getSoft([imgsize,imgsize],xb,yb)
                    soft = torch.Tensor(soft).unsqueeze(0)
                    mask = (torch.ones(mask.shape)-mask)*soft + mask
                hair_A = (A/2+0.5) * mask.repeat(int(input_nc/output_nc),1,1) * 2 - 1
                item['hair_A'] = hair_A
                item['mask'] = mask
                if self.opt.bg_local:
                    bgdir = self.opt.bg_dir
                    bgpath = os.path.join(bgdir,basen[:-4]+'.png')
                    im_bg = Image.open(bgpath)
                    mask2 = transforms.ToTensor()(im_bg) # mask out background
                    mask2 = (mask2 >= 0.5).float()
                    hair_A = (A/2+0.5) * mask.repeat(int(input_nc/output_nc),1,1) * mask2.repeat(int(input_nc/output_nc),1,1) * 2 - 1
                    bg_A = (A/2+0.5) * (torch.ones(mask2.shape)-mask2).repeat(int(input_nc/output_nc),1,1) * 2 - 1
                    item['hair_A'] = hair_A
                    item['bg_A'] = bg_A
                    item['mask'] = mask
                    item['mask2'] = mask2

        return item

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
