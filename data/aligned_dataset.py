import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import cv2
import csv

def getfeats(featpath):
	trans_points = np.empty([5,2],dtype=np.int64) 
	with open(featpath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for ind,row in enumerate(reader):
			trans_points[ind,:] = row
	return trans_points

def tocv2(ts):
    img = (ts.numpy()/2+0.5)*255
    img = img.astype('uint8')
    img = np.transpose(img,(1,2,0))
    img = img[:,:,::-1]#rgb->bgr
    return img

def dt(img):
    if(img.shape[2]==3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #convert to BW
    ret1,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret2,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    dt1 = cv2.distanceTransform(thresh1,cv2.DIST_L2,5)
    dt2 = cv2.distanceTransform(thresh2,cv2.DIST_L2,5)
    dt1 = dt1/dt1.max()#->[0,1]
    dt2 = dt2/dt2.max()
    return dt1, dt2

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

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        imglist = 'datasets/apdrawing_list/%s/%s.txt' % (opt.phase, opt.dataroot)
        if os.path.exists(imglist):
            lines = open(imglist, 'r').read().splitlines()
            lines = sorted(lines)
            self.AB_paths = [line.split()[0] for line in lines]
            if len(lines[0].split()) == 2:
                self.B_paths = [line.split()[1] for line in lines]
        else:
            self.dir_AB = os.path.join(opt.dataroot, opt.phase)
            self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        if w/h == 2:
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        else: # if w/h != 2, need B_paths
            A = AB.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B = Image.open(self.B_paths[index]).convert('RGB')
            B = B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]#C,H,W
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        flipped = False
        if (not self.opt.no_flip) and random.random() < 0.5:
            flipped = True
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        
        item = {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

        if self.opt.use_local:
            regions = ['eyel','eyer','nose','mouth']
            basen = os.path.basename(AB_path)[:-4]+'.txt'
            if self.opt.region_enm in [0,1]:
                featdir = self.opt.lm_dir
                featpath = os.path.join(featdir,basen)
                feats = getfeats(featpath)
                if flipped:
                    for i in range(5):
                        feats[i,0] = self.opt.fineSize - feats[i,0] - 1
                    tmp = [feats[0,0],feats[0,1]]
                    feats[0,:] = [feats[1,0],feats[1,1]]
                    feats[1,:] = tmp
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
                    item[regions[i]+'_A'] = A[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)]
                    item[regions[i]+'_B'] = B[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)]
                    if self.opt.soft_border:
                        item[regions[i]+'_A'] = item[regions[i]+'_A'] * soft_border_mask4[i].repeat(int(input_nc/output_nc),1,1)
                        item[regions[i]+'_B'] = item[regions[i]+'_B'] * soft_border_mask4[i]
            if self.opt.compactmask:
                cmasks0 = []
                cmasks = []
                for i in range(4):
                    if flipped and i in [0,1]:
                        cmaskpath = os.path.join(self.opt.cmask_dir,regions[1-i],basen[:-4]+'.png')
                    else:
                        cmaskpath = os.path.join(self.opt.cmask_dir,regions[i],basen[:-4]+'.png')
                    im_cmask = Image.open(cmaskpath)
                    cmask0 = transforms.ToTensor()(im_cmask)
                    if flipped:
                        cmask0 = cmask0.index_select(2, idx)
                    if output_nc == 1 and cmask0.shape[0] == 3:
                        tmp = cmask0[0, ...] * 0.299 + cmask0[1, ...] * 0.587 + cmask0[2, ...] * 0.114
                        cmask0 = tmp.unsqueeze(0)
                    cmask0 = (cmask0 >= 0.5).float()
                    cmasks0.append(cmask0)
                    cmask = cmask0.clone()
                    if self.opt.region_enm in [0,1]:
                        cmask = cmask[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)]
                    elif self.opt.region_enm in [2]: # need to multiply cmask
                        item[regions[i]+'_A'] = (A/2+0.5) * cmask * 2 - 1
                        item[regions[i]+'_B'] = (B/2+0.5) * cmask * 2 - 1
                    cmasks.append(cmask)
                item['cmaskel'] = cmasks[0]
                item['cmasker'] = cmasks[1]
                item['cmask'] = cmasks[2]
                item['cmaskmo'] = cmasks[3]
            if self.opt.hair_local:
                mask = torch.ones(B.shape)
                if self.opt.region_enm == 0:
                    for i in range(4):
                        mask[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)] = 0
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
                            xbi = [center[i,0]-rws[i]//2, center[i,0]+rws[i]//2-1]
                            ybi = [center[i,1]-rhs[i]//2, center[i,1]+rhs[i]//2-1]
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
                elif self.opt.region_enm == 1:
                    for i in range(4):
                        cmask0 = cmasks0[i]
                        rec = torch.zeros(B.shape)
                        rec[:,center[i,1]-rhs[i]//2:center[i,1]+rhs[i]//2,center[i,0]-rws[i]//2:center[i,0]+rws[i]//2] = 1
                        mask = mask * (torch.ones(B.shape) - cmask0 * rec)
                elif self.opt.region_enm == 2:
                    for i in range(4):
                        cmask0 = cmasks0[i]
                        mask = mask * (torch.ones(B.shape) - cmask0)
                hair_A = (A/2+0.5) * mask.repeat(int(input_nc/output_nc),1,1) * 2 - 1
                hair_B = (B/2+0.5) * mask * 2 - 1
                item['hair_A'] = hair_A
                item['hair_B'] = hair_B
                item['mask'] = mask # mask out eyes, nose, mouth
                if self.opt.bg_local:
                    bgdir = self.opt.bg_dir
                    bgpath = os.path.join(bgdir,basen[:-4]+'.png')
                    im_bg = Image.open(bgpath)
                    mask2 = transforms.ToTensor()(im_bg) # mask out background
                    if flipped:
                        mask2 = mask2.index_select(2, idx)
                    mask2 = (mask2 >= 0.5).float()
                    hair_A = (A/2+0.5) * mask.repeat(int(input_nc/output_nc),1,1) * mask2.repeat(int(input_nc/output_nc),1,1) * 2 - 1
                    hair_B = (B/2+0.5) * mask * mask2 * 2 - 1
                    bg_A = (A/2+0.5) * (torch.ones(mask2.shape)-mask2).repeat(int(input_nc/output_nc),1,1) * 2 - 1
                    bg_B = (B/2+0.5) * (torch.ones(mask2.shape)-mask2) * 2 - 1
                    item['hair_A'] = hair_A
                    item['hair_B'] = hair_B
                    item['bg_A'] = bg_A
                    item['bg_B'] = bg_B
                    item['mask'] = mask
                    item['mask2'] = mask2
        
        if (self.opt.isTrain and self.opt.chamfer_loss):
            if self.opt.which_direction == 'AtoB':
                img = tocv2(B)
            else:
                img = tocv2(A)
            dt1, dt2 = dt(img)
            dt1 = torch.from_numpy(dt1)
            dt2 = torch.from_numpy(dt2)
            dt1 = dt1.unsqueeze(0)
            dt2 = dt2.unsqueeze(0)
            item['dt1gt'] = dt1
            item['dt2gt'] = dt2
        
        if self.opt.isTrain and self.opt.emphasis_conti_face:
            face_mask_path = os.path.join(self.opt.facemask_dir,basen[:-4]+'.png')
            face_mask = Image.open(face_mask_path)
            face_mask = transforms.ToTensor()(face_mask) # [0,1]
            if flipped:
                face_mask = face_mask.index_select(2, idx)
            item['face_mask'] = face_mask

        return item

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
