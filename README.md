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
python3 ./train.py --dataroot path_to_dir_for_reflection_synthesis/ --name reflection_synthesis --gpu_ids 0 --save_epoch_freq 1 --batchSize 10
```
or you can directly:
``` bash 
bash ./synthesis_train.sh
```
* To test the synthesis model:
``` bash
python3 ./test.py --dataroot path_to_dir_for_synthesis/ --name reflection_synthesis --gpu_ids 0 --which_epoch 130 --how_many 1
```
or you can directly:
``` bash 
bash ./synthesis_test.sh
```
[Here](https://www.baidu.com/) is the pre-trained model. And to generate the three types of reflection images, you can use [these original images](https://www.baidu.com/) from [perceptual-reflection-removal](https://github.com/ceciliavision/perceptual-reflection-removal).
## Reflection Removal
``` bash
cd ./Removal
```
* Constrcut these new folders for training and testing

  training set: trainA(contains the reflection ground truth.), trainB(contains the transmission ground truth), trainC(contains the images which have the reflection to remove.), trainW(contains the alpha blending mask ground truth.)
  
  testing set: trainB(contains the transmission ground truth), trainC(contains the reflection images to be remove.)
