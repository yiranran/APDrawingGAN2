set -ex
python train.py --dataroot dataset/AB_140_aug3_H_hm2 --name apdrawinggan++_1 --model apdrawingpp_style --use_resnet --netG resnet_9blocks --continue_train --continuity_loss --lambda_continuity 40.0 --gpu_ids 0 --gpu_ids_p 1 --display_env apdrawinggan++_1 --niter 200 --niter_decay 0 --lr 0.0001 --batch_size 1 --emphasis_conti_face --auxiliary_root auxiliaryeye2o

