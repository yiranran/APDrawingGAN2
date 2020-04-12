## Preprocessing steps

Both training and testing images need: 

- align to 512x512
- facial landmarks
- mask for eyes,nose,mouth,background

Training images additionally need:

- mask for face region


### 1. Align, resize, crop images to 512x512, and get facial landmarks

All training and testing images in our model are aligned using facial landmarks. And landmarks after alignment are needed in our code.

- First, 5 facial landmark for a face photo need to be detected (we detect using [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)(MTCNNv1)).

- Then, we provide a matlab function in `face_align_512.m` to align, resize and crop face photos (and corresponding drawings) to 512x512.Call this function in MATLAB to align the image to 512x512.
For example, for `img_1701.jpg` in `example` dir, 5 detected facial landmark is saved in `example/img_1701_facial5point.mat`. Call following in MATLAB:
```bash
load('example/img_1701_facial5point.mat');
[trans_img,trans_facial5point]=face_align_512('example/img_1701.jpg',facial5point,'example');
```

This will align the image, and output aligned image + transformed facial landmark (in txt format) in `example` folder.
See `face_align_512.m` for more instructions.

The saved transformed facial landmark need to be copied to `dataset/landmark/`, and has the **same filename** with aligned face photos (e.g. `dataset/data/test_single/31.png` should have landmark file `dataset/landmark/31.txt`).

### 2. Prepare background masks

In our work, background mask is segmented by method in
"Automatic Portrait Segmentation for Image Stylization"
Xiaoyong Shen, Aaron Hertzmann, Jiaya Jia, Sylvain Paris, Brian Price, Eli Shechtman, Ian Sachs. Computer Graphics Forum, 35(2)(Proc. Eurographics), 2016.

We use code in http://xiaoyongshen.me/webpage_portrait/index.html to detect background masks for aligned face photos.
An example background mask is shown in `example/img_1701_aligned_bgmask.png`.

The background masks need to be copied to `dataset/mask/bg/`, and has the **same filename** with aligned face photos (e.g. `dataset/data/test_single/31.png` should have background mask `dataset/mask/bg/31.png`)  

### 3. Prepare eyes/nose/mouth masks

We use dlib to extract 68 landmarks for aligned face photos, and use these landmarks to get masks for local regions.
See an example in `get_partmask.py`, the eyes, nose, mouth masks for `example/img_1701_aligned.png` are `example/img_1701_aligned_[part]mask.png`, where part is in [eyel,eyer,nose,mouth].

The part masks need to be copied to `dataset/mask/[part]/`, and has the **same filename** with aligned face photos.

### 4. (For training) Prepare face masks

We use the face parsing net in https://github.com/cientgu/Mask_Guided_Portrait_Editing to detect face region.
The face parsing net will label each face into 11 classes, the 0 is for background, 10 is for hair, and the 1~9 are face regions.
An example face mask is shown in `example/img_1701_aligned_facemask.png`.

The face masks need to be copied to `dataset/mask/face/`, and has the **same filename** with aligned face photos.

### 5. (For training) Combine A and B

We provide a python script to generate training data in the form of pairs of images {A,B}, i.e. pairs {face photo, drawing}. This script will concatenate each pair of images horizontally into one single image. Then we can learn to translate A to B:

Create folder `/path/to/data` with subfolders `A` and `B`. `A` and `B` should each have their own subfolders `train`, `test`, etc. In `/path/to/data/A/train`, put training face photos. In `/path/to/data/B/train`, put the corresponding artist drawings. Repeat same for `test`.

Corresponding images in a pair {A,B} must both be images after aligning and of size 512x512, and have the same filename, e.g., `/path/to/data/A/train/1.png` is considered to correspond to `/path/to/data/B/train/1.png`.

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.