# Object Detection Challenge

## Prerequisite
- Operation System: Ubuntu 20.04
- Package Manager: Anaconda

## Yolov3-tensorflow2
1. clone the code
```
git clone https://github.com/zzh8829/yolov3-tf2.git
```
2. create a GPU environment
```
conda env create -f conda-gpu.yml
conda activiate yolov3-tf2-gpu
pip install -r requirements-gpu.txt
```
3. Nvidia Driver (for GPU)
On Ubuntu 20.04, directly update the nvidia-driver-460 in driver updater
use command to check your nvidia devices
```
ubuntu-drivers devices
```
But for this repo, use nvidia-driver-430 instead

4. Convert pre-trained Darknet weights
```
# Yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

## Open Image Data v6 Download
1. clone the code
```
git clone https://github.com/pythonlessons/OIDv4_ToolKit
```
2. install dependency
```
pip install -r requirements.txt
```
3. download class "Apple"
```
python main.py downloader --classes Apple --type_csv validation
```
An OID folder will be created with two sub folder "csv_folder" and "Dataset" where all the downloaded images are located. Ignore the promot "Error", Enter "y" for download the missing file
4. support OIDv6 download
  - open https://storage.googleapis.com/openimages/web/download.html, click "V6" button in the first line and then scroll down to "Download the annotations and metadata" and click "Train" button to download train csv file in "Boxes" row.
  - After downloading, put the "oidv6-train-annotations-bbox.csv" into the folder of "OID/scv_folder", and change the name to "train-annotation-bbox.csv" (you also can edit the code in main.py if you don't want to rename the csv file)
  - repeat previous steps for downloading the "test" and "validation" annotation bbox
  - now the csv_folder contains 4 files including the "class-description-boxable.csv"
  - use the same command as usual.
5. download images
  - create classes.txt (specified your interested classes)
  - download all the images of interested classes with a limited number
  ```
  python main.py downloader --classes classes.txt --type_csv train --limit 1500
  ```

## Convert OID image to tfrecord
1. clone the code, and make replace 'app' with 'compat.v1' in the code
```
git clone https://github.com/zamblauskas/oidv4-toolkit-tfrecord-generator.git
```
2. convert oid image dataset to tfrecord
```
python generate-tfrecord.py \
--classes_file=./OIDv4_ToolKit/classes.txt \
--class_descriptions_file=./OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv \
--annotations_file=./OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv \
--images_dir=./OIDv4_ToolKit/OID/Dataset/train \
--output_file=./dataset/oid/train.tfrecord
```

## Training with your own interested classes
- with Transfer Learning
Origanl pretrained yolov3 has 80 classes, here we learn 9 classes.
```
python train.py \
	--dataset ./dataset/oid/train.tfrecord \
	--val_dataset ./dataset/oid/val.tfrecord \
	--classes ./dataset/oid/classes.names \
	--num_classes 9 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 10 \
	--weights ./checkpoints/yolov3.tf \
	--weights_num_classes 80
```
- Training from random weights
Training from scrath is very difficult to converge. The original paper trained darknet on imagenet before training the whole network as well.
```
python train.py \
	--dataset ./dataset/oid/train.tfrecord \
	--val_dataset ./dataset/oid/val.tfrecord \
	--classes ./dataset/oid/classes.names \
	--num_classes 9 \
	--mode fit --transfer none \
	--batch_size 16 \
	--epochs 10 \
```

## Inference
- detect from images
```
python detect.py \
	--classes ./data/oidv6.names \
	--num_classes 9 \
	--weights ./checkpoints/yolov3_train_5.tf \
	--image ./data/street.jpg
```
- detect from validation set
```
python detect.py \
	--classes ./data/oidv6.names \
	--num_classes 9 \
	--weights ./checkpoints/yolov3_train_5.tf \
	--tfrecord ./data/oidv6_val.tfrecord
```



## References
- [Train YOLOv3 with OpenImagesv4](https://github.com/WyattAutomation/Train-YOLOv3-with-OpenImagesV4)
