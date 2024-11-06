# ClotScape: Enhancing CT Scan PAA Segmentation

ClotScape provides a pre-processing pipeline to enhance the accuracy of popliteal artery aneurysm (PAA) segmentation in knee CT scans. This tool improves segmentation by addressing challenges like noise, intensity variation, and artifact interference.

## Pipeline Overview
1. **Denoising**: Gaussian filtering to reduce noise.
2. **Normalization**: CLAHE normalization for contrast enhancement.
3. **Edge Detection**: Sobel edge detection to highlight boundaries.

## Key Results
- **Baseline Segmentation**: Default configuration shows limited accuracy.
- **Enhanced Segmentation**: Pre-processed images achieve significantly better mask alignment with PAA boundaries.

Research and results can be accessed through the [Notebooks](https://github.com/punitarani/clotscape/tree/747b620ff60a73c87b8ecbec705d2bef6cd02693/notebooks)
[Segmentation Demo](https://github.com/punitarani/clotscape/blob/747b620ff60a73c87b8ecbec705d2bef6cd02693/notebooks/segment.ipynb)
