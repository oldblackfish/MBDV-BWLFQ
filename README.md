**Official PyTorch code for the ICME2025 paper "Mamba-Based Blind Stitched Wide Field of View Light Field Image Quality Assessment via Dual-Viewport Sampling". Please refer to the [paper](https://ieeexplore.ieee.org/document/11209926) for details.**

### Poster
![image](https://github.com/oldblackfish/MBDV-BWLFQ/blob/main/fig/ICME2025_Poster.png)

**Note: First, we convert the dataset into H5 files using MATLAB. Then, we train and test the model in Python.**

### Generate Dataset in MATLAB
Convert the dataset into h5 files, and then put them into './Datasets/WLFI_224x224_dual/':
```
 ./MBDV-BWLFQ/Datasets/Generateh5_for_Dataset.m
```
    
### Train
Train the model using the following command:
```
python Train.py
```

### Test Overall Performance
Test the overall performance using the following command:
```
python Test.py
```

### Citation
Please cite the following paper if you use this repository in your reseach.
```
@INPROCEEDINGS{11209926,
  author={Zhou, Rui and Jiang, Gangyi and Zhu, Linwei and Chen, Yeyao and Cui, Yueli and Luo, Ting and Xu, Haiyong},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Mamba-Based Blind Stitched Wide Field of View Light Field Image Quality Assessment via Dual-Viewport Sampling}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Measurement;Image quality;Training;Benchmark testing;Distortion;Feature extraction;Multitasking;Light fields;Hardware;Quality assessment;Blind image quality assessment;stitched wide field of view light field image (WLFI);Mamba;dual-viewport},
  doi={10.1109/ICME59968.2025.11209926}}
```
### Contact
For any questions, feel free to contact: 2211100079@nbu.edu.cn or zhourui0628@163.com
