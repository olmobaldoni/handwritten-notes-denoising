# Handwritten-notes Denoising

This project provides tools to accelerate the collection of data needed to address mathematical equation detection and optical character recognition (OCR) in handwritten notes. Handwritten notes pose several difficulties for automatic extraction of text and equations, mainly because of the inherent noise created when notes are taken on lined or squared sheets, rather than on a blank sheet of paper. Our goal is to have a Convolutional Neural Network (CNN) trained to remove square grids from handwritten notes and to obtain a text-only, noise-free image.

## Synthetic Dataset for Denoising Handwritten Notes

The most significant work has been the generation of the Synthetic Dataset needed to train UNet to recognize and eliminate squares or rows from non-white sheets. The Dataset consists of 132,000 synthetic images obtained by a Data Augmentation process. For its creation, we collected 3,300 images of handwritten notes on blank sheets and 70 templates of different types of squares or rows as background.

## Image Cropping and Warping

We present a simple approach to crop and warp images of hand-taken notes from photos. The implementation involves detecting the edges of the document, identifying the outline representing the paper, and applying a perspective transformation to obtain a top-down view of the document.

## Architectures and Results

We chose UNet as the network to implement the task of denoising grids and lines. UNet is a convolutional neural network (CNN) architecture commonly used for segmentation problems, where the goal is to identify and separate different regions or objects within an image. The main reason we chose it is the downsampling and upsampling protocol; the architecture was designed to effectively downsample and upsample information in images.

## CBIR for Handwritten Notes

We have implemented an image retrieval system to identify notes that require background denoising, from those that are already suitable for the text recognition task. In other words, it allows us to automatically identify images that require processing. The system is also capable of grouping notes from the same classes of notes. In this way, we can retrieve an arbitrary number of similar images from a single image of notes.

## Future Developments

Our proposed work could continue in the following ways:

- [ ] The training dataset could be further augmented with more variations to better match real images.
- [ ] Instead of training UNet from scratch, fine-tuning a pretrained model could boost performance.
- [ ] Alternative learning rate schedulers like cyclical or one policies could be explored.
- [ ] More advanced metrics beyond MSE/SSIM would better evaluate the similarity of denoised images to ground truth.
- [ ] Using Vision Transformer (ViT) or DINO for image embeddings may improve the retrieval system.
- [ ] The retrieval system could be extended to find similarity between crops rather than full images.


## Aknowledgements

|AUTHORs|CONTACTs|GITHUBs|
|-|-|-|
|Olmo Baldoni|[325524@studenti.unimore.it](mailto:325524@studenti.unimore.it)|[olmobaldoni](https://github.com/olmobaldoni)|
|Cristian Bellucci|[322906@studenti.unimore.it](mailto:322906@studenti.unimore.it)|[cleb98](https://github.com/cleb98)|
|Danilo Caputo|[212017@studenti.unimore.it](mailto:246019@studenti.unimore.it)|[RiccardoSanti092](https://github.com/IloDan)|

---

## License

MIT
