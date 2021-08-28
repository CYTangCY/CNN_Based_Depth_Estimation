# CNN_Based_Depth_Estimation
Use flowdep_V1 and 2Dgray_image to predict depth with IMU and Camera.

## Dependencies

### Run&Train

* numba
* torch
* torchvision
* numpy
* scipy

### Visualize

* opencv
* cmapy
* matplotlib

## Usage
### To Train the model
you need to download DataSet from Kitti DepthEstimation(for Depth map) and rawDataset(for IMU_DATA and RawImage to calculate opticalflow)

```bash
python Train_module.py
```

To test the model, you need to download the dataset from here(Pre-Train model and Crop DataSet):
https://drive.google.com/drive/folders/16kIMDawuQ6MVet9yxr4shkWg71a4Skpm?usp=sharing

```bash
python Visualize_result.py
```
## Configuration
You need to change the PATH of DATAset and model in `Train_module.py` and `Visualize_result`
