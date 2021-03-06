# Labelling_utilities

Simple utilities which can be used for labelling images for object detection and instance segmentation. 

### 1. 2D bounding boxes using instance segmentation

 Object detection is crucial for detecting and localizing dynamic objects such as Cars, Pedestrians in images.
 But some dataset may not necessarily provide ground truth for bounding boxes, but most often they do have other ground truth labels like instance segmentation.
 This [notebook](./bb_instance.ipynb) simply aims at automating the task of labelling images with bounding box using instance segmentation.
 
   a. **Input-**   KITTI - 360 dataset - (RGB images + instance segmentation)
 
   b. **Output-**   Bounding boxes for dynamic objects like cars and pedestrians
 
 

### 2. Generating instance segmentation in camera images using Lidar ground truth
 
 Carla simulator offers, many ground truth generation options, but still there is no support for instance segmentaion for camera images. But they do provide instance segmentaion for Lidar. Using Lidar projeciton on images, instance segmentation for camera images are generated.

 a. **Input-** Sematic Lidar and Camera sensor attached to ego vehicle
 
 b. **Output-** instance segmentation for camera images
 
 Please make sure carla server, before running the starting [carla client](./lidar_to_camera_instance_map.py)
 
 
 **Input Image:**
 
 ![input image](lidar_to_camera_instance_map/279.png)
 
 
 **Corresponding instance map:**
 
 ![corresponding_instance_segmentation](lidar_to_camera_instance_map/279_instance.png)
 
 
## Acknowledgement :pray:
- [KITTI 360 Dataset Team](http://www.cvlibs.net/datasets/kitti-360/)
- [Carla Simualtor Team](https://carla.org/)




  

