set -ex
python test.py --dataroot dataset/AB_140_aug3_H_hm2 --name apdrawinggan++_author --model apdrawingpp_style --use_resnet --netG resnet_9blocks --which_epoch 150 --how_many 1000 --gpu_ids 0 --gpu_ids_p 0 --imagefolder images-apd70
