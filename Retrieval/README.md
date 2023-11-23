# CBIR for Handwritten Notes

Content-Based Image Retrieval (CBIR) system for handwritten notes using convolutional neural networks (CNNs) with Regional Maximum Activation of Convolution (RMAC) vector descriptors. The CBIR system aims to simplify the denoising process by retrieving similar handwritten notes based on visual features, focusing on variations in the grid patterns of the paper as defining categories. In this context, different backbones are explored and employed for effective comparison.

## Datasets for Image Retrieval

We evaluated the RMAC-based retrieval system on three distinct datasets constructed using
different methods: raw unprocessed scanned images of notes from various subjects and synthetic
images generated for UNet network training.
- **Dataset 1** ("db_0") consists of 120 unprocessed note images, with 20 images per class,
from six different subjects.
- **Dataset 2** ("db_1") comprises 120 images, including 60 without gridlines and 60 artificially
generated images. These generated images are similar to the ones used for training the
UNet network in image denoising. There are no associations between the images across
the two classes.
- **Dataset 3** ("db_2") includes 200 images, with 20 images without gridlines and 180 images
generated from the previous set using nine different grid styles. These generated grid
images are of the same type as the ones used for training the UNet network in image
denoising. There are associations between the non-grid and generated grid images across
the ten classes, with 20 images per class.

## Image Retrieval Pipeline

![CBIR Pipeline](./assets/CBIR_architecture.png)

1. Given a query image, it is first preprocessed and converted into a tensor.
2. The query image tensor is passed through the pre-trained CNN feature extractor to obtain
the convolutional activation maps.
3. Regional maximum activation of convolutions (R-MAC) is applied on the activation maps
to obtain a regional feature vector for each image region at multiple scales.
4. The regional feature vectors are l2-normalized, summed and l2-normalized again to obtain
a single global image descriptor vector.
5. The query descriptor is compared against the precomputed database descriptors using
cosine similarity to retrieve the top k most similar images.

## Experiments and Results

The experiments evaluated multiple backbone architectures, including VGG16, VGG19, DenseNet, UNet trained on image denoising, and three UNet models with Kaiming initialization and random weights. The experiments were conducted with a 40% overlap ratio and 6 sampling scales. Each image served as a query for the CBIR system, and the Average Precision (AP) was computed for each query image. The mean Average Precision (mAP) was calculated across all query images for each backbone architecture on the respective dataset. The mAP@K values for K = 3, 5, 10, and 20 were reported to provide a comprehensive view of the system's performance in different top-K retrievals. The results for each backbone architecture on various datasets are presented in the following section.

- ### mAP on db\_0

| Model         | k = 3  | k = 5  | k = 10 | k = 20 |
|---------------|--------|--------|--------|--------|
| VGG16         | 0.997  | 0.997  | 0.993  | 0.951  |
| VGG19         | 0.997  | 0.998  | 0.994  | 0.930  |
| DenseNet      | 1.000  | 1.000  | 0.998  | 0.923  |
| Trained UNet  | 0.983  | 0.975  | 0.864  | 0.680  |
| Kaiming UNet 0| 1.000  | 0.995  | 0.989  | 0.923  |
| Kaiming UNet 1| 1.000  | 1.000  | 0.998  | 0.945  |
| Kaiming UNet 2| 1.000  | 0.998  | 0.996  | 0.935  |

- ### mAP on db\_1

| Model         | k = 3  | k = 5  | k = 10 | k = 20 |
|---------------|--------|--------|--------|--------|
| VGG16         | 0.972  | 0.955  | 0.927  | 0.894  |
| VGG19         | 0.950  | 0.937  | 0.905  | 0.876  |
| DenseNet      | 0.972  | 0.958  | 0.932  | 0.894  |
| Trained UNet  | 0.908  | 0.858  | 0.806  | 0.758  |
| Kaiming UNet 0| 0.939  | 0.903  | 0.854  | 0.768  |
| Kaiming UNet 1| 0.939  | 0.912  | 0.864  | 0.785  |
| Kaiming UNet 2| 0.906  | 0.868  | 0.837  | 0.740  |

- ### mAP on db\_2

| Model         | k = 3  | k = 5  | k = 10 | k = 20 |
|---------------|--------|--------|--------|--------|
| VGG16         | 0.675  | 0.636  | 0.604  | 0.548  |
| VGG19         | 0.717  | 0.711  | 0.674  | 0.594  |
| DenseNet      | 0.730  | 0.724  | 0.677  | 0.613  |
| Trained UNet  | 0.467  | 0.360  | 0.317  | 0.302  |
| Kaiming UNet 0| 0.467  | 0.360  | 0.354  | 0.348  |
| Kaiming UNet 1| 0.467  | 0.370  | 0.371  | 0.375  |
| Kaiming UNet 2| 0.467  | 0.360  | 0.347  | 0.336  |

- Overall, the pretrained CNNs (VGG16, VGG19, DenseNet) consistently achieve the best performance on all three datasets, due to their ability to extract powerful
- The UNet trained for the denoising task performs the worst as it overfits to that task and cannot generalize well for the retrieval task.


## Install Dependencies

```console
handwritten-notes-denoising/retrieval pip install -r requirements.txt
```

## Download Datasets

Once the dataset is downloaded, add the `data` folder to the respective `db` folder in the cloned repository

- [db_0](https://archive.org/compress/db_0_20231123/formats=ITEM%20TILE,JPEG,ARCHIVE%20BITTORRENT,METADATA)
- [db_1](https://archive.org/compress/db_1_20231123/formats=ITEM%20TILE,PNG,ARCHIVE%20BITTORRENT,METADATA)
- [db_2](https://archive.org/compress/db_2_20231123/formats=ITEM%20TILE,PNG,ARCHIVE%20BITTORRENT,METADATA)

## Run

- **Demo retrieve images:** Run jupyter notebook

- **mAP evaluation:** Run python script
```console
handwritten-notes-denoising/retrieval python eval_db_<>.py
```


## References

[1] Konstantin Schall, Kai Uwe Barthel, Nico Hezel, Klaus Jung. *GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval*. [arXiv:2111.13122](https://arxiv.org/abs/2111.13122)

[2] Giorgos Tolias, Ronan Sicre, Hervé Jégou. *Particular object retrieval with integral max-pooling of CNN activations*. [arXiv:1511.05879](https://arxiv.org/abs/1511.05879)

[3] Yang Li, Yulong Xu, Jiabao Wang, Zhuang Miao, Yafei Zhang. *MS-RMAC: Multiscale Regional Maximum Activation of Convolutions for Image Retrieval*. IEEE Signal Processing Letters, Vol. PP, pp. 1-1, Feb 2017. [DOI: 10.1109/LSP.2017.2665522](https://doi.org/10.1109/LSP.2017.2665522)
