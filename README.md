# Single-Image-Reflection-Removal-Beyond-Linearity
## Paper
[Single Image Reflection Removal Beyond Linearity](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wen_Single_Image_Reflection_Removal_Beyond_Linearity_CVPR_2019_paper.pdf).

Qiang Wen, Yinjie Tan, Jing Qin, Wenxi Liu, Guoqiang Han, and Shengfeng He*
## Requirement
- Python 3.5
- PIL
- OpenCV-Python
- Numpy
- Pytorch 0.4.0
- Ubuntu 16.04 LTS
## Reflection Synthesis
``` bash
cd ./Synthesis
```
* Constrcut these new folders for training and testing

  training set: trainA, trainB, trainC(contains real-world reflection images for adversarial loss.)
  
  testing set: testA(contains the images to be used as reflection.), testB(contains the images to be used as transmission.)
* To train the synthesis model:
``` bash
python3 ./train.py --dataroot path_to_dir_for_reflection_synthesis/ --gpu_ids 0 --save_epoch_freq 1 --batchSize 10
```
or you can directly:
``` bash 
bash ./synthesis_train.sh
```
* To test the synthesis model:
``` bash
python3 ./test.py --dataroot path_to_dir_for_synthesis/ --gpu_ids 0 --which_epoch 130 --how_many 1
```
or you can directly:
``` bash 
bash ./synthesis_test.sh
```
[Here](https://drive.google.com/file/d/1il1jvCK2fb8g9NDA66bbazP5TwTbjdGU/view?usp=sharing) is the pre-trained model. And to generate the three types of reflection images, you can use [these original images](https://drive.google.com/file/d/1DNKj4YM4OjE_brXayf0hmqYQb3M6OqK7/view?usp=sharing) which are from [perceptual-reflection-removal](https://github.com/ceciliavision/perceptual-reflection-removal).

Due to the **copyright**, the real reflection images are not released here.
## Reflection Removal
``` bash
cd ./Removal
```
* Constrcut these new folders for training and testing

  training set: trainA(contains the reflection ground truth.), trainB(contains the transmission ground truth), trainC(contains the images which have the reflection to remove.), trainW(contains the alpha blending mask ground truth.)
  
  testing set: testB(contains the transmission ground truth), testC(contains the images which have the reflection to remove.)
* To train the removal model:
``` bash
python3 ./train.py --dataroot path_to_dir_for_reflection_removal/ --gpu_ids 0 --save_epoch_freq 1 --batchSize 5 --which_type focused
```
or you can directly:
``` bash 
bash ./removal_train.sh
```
* To test the removal model:
``` bash
python3 ./test.py --dataroot path_to_dir_for_reflection_removal/ --which_type focused --which_epoch 130 --how_many 1
```
or you can directly:
``` bash 
bash ./removal_test.sh
```
Here are the [pre-trained models](https://drive.google.com/file/d/1WbfcVGTu91yZP26o8DKCrdQhyXpHt6Fo/view?usp=sharing) which are trained on the three types of synthetic dataset.

Here are the synthetic [training set](https://drive.google.com/drive/folders/1cPoeIrRU0qxA8cvkL7x2pOIJb7KLF9aM?usp=sharing) and [testing set](https://drive.google.com/file/d/1JBXAdAXgFreTq3zmP6NvuEp2CqK3o8ng/view?usp=sharing) for reflection removal.

To evaluate on other datasets, please finetune the pre-trained models or re-train a new model on the specific training set.
## Acknowledgments
Part of the code is based upon [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
## Citation
```
@InProceedings{Wen_2019_CVPR,
  author = {Wen, Qiang and Tan, Yinjie and Qin, Jing and Liu, Wenxi and Han, Guoqiang and He, Shengfeng},
  title = {Single Image Reflection Removal Beyond Linearity},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```
