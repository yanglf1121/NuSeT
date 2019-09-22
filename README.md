# NuSeT
NuSeT: A Deep Learning Tool for Reliably Separating and Analyzing Crowded Cells

## To run the code
The following packages are needed for NuSeT to run:
1. tensorflow (*pip3 install tensorflow / pip3 install tensorflow-gpu*)
2. PIL (*pip3 install Pillow*)
3. numpy (*pip3 install numpy*)
4. scikit-image (*pip3 install scikit-image*)
5. tqdm (*pip3 install tqdm*)

**NuSeT works better for moderate sized nuclei/cells, please adjust the 'resize ratio' in 'Configuration' section under 'Predicting' module. For optimal nuclei/cells sizes, please refer to images under 'sample_image' folder.**

After finishing installing packages, download the 2 weight files from google drive:(https://drive.google.com/file/d/1fcs1F2lGPX0ejzEGPZ63YNF3AmUbdBcM/view?usp=sharing
https://drive.google.com/file/d/1hythQfvD6kbaUClAPY96nHcXB7RXVmBx/view?usp=sharing), move those files to Network/ folder. 
Then navigate to the root folder of this repo (NuSeT/), in the command line run *python3 NuSeT.py*.
<p align="center">
<img src="https://github.com/yanglf1121/NuSeT/blob/master/GUI_samples/GUI.png" alt="alt text" width="200">
</p>
<p align="center">
<img src="https://github.com/yanglf1121/NuSeT/blob/master/GUI_samples/t_configure.png" alt="alt text" width="200">
</p>
<p align="center">
<img src="https://github.com/yanglf1121/NuSeT/blob/master/GUI_samples/training.png" alt="alt text" width="400">
</p>
<p align="center">
<img src="https://github.com/yanglf1121/NuSeT/blob/master/GUI_samples/p_configure.png" alt="alt text" width="200">
</p>
<p align="center">
<img src="https://github.com/yanglf1121/NuSeT/blob/master/GUI_samples/seg-results.png" alt="alt text" width="400">
</p>
For the detailed user-guide, please see our paper: https://www.biorxiv.org/content/10.1101/749754v1

## The motivation for this work
Tools for segmenting fluorescent nuclei need to address multiple features and limitations of biological images. Typical issues and limitations include:

1)	Boundary assignment ambiguity: biological samples frequently have very high cell density with significant overlap between objects. 
2)	Signal intensity variation: Within one image, the signal can vary within each nucleus (e.g. due to different compaction states of the DNA in heterochromatin vs. euchromatin) and across nuclei (e.g. due to cell to cell differences in nuclear protein expression levels and differences in staining efficiency). 
3)	Non-cellular artifacts and contaminants: Fluorescence microscopy samples are often contaminated with auto-fluorescent cell debris as well as non-cellular artifacts. 
4)	Low signal to noise ratios (SNRs): Low SNRs typically result from lower expression levels of fluorescent targets and/or high background signal, such as sample auto fluorescence. 

## The highlights for this work
This work took the advantages of two state-of-the-art cell segmentation models, **U-Net** and **Mask-RCNN**. This work also incorporated other algorithms to specifically address issues cell biologists may encounter during imaging. The improvements include:

1. Fusing U-Net parallely with Region Proposal Network (RPN), following a watershed transform to achieve accurate cell-boundary assignment in dense environment.
2. Foreground normalization to improve detection on sample paration artifacts and signal variations
3. Sythetic images to further improve detection on sample paration artifacts and cell boundary assignment.
4. Graphic user interface for using pre-trained NuSeT models, and for training new models using custom training data.

## Results using pre-trainined NuSeT model (on Kaggle 2018 data science bowl and images from our lab)
<p align="center">
<img src="https://github.com/yanglf1121/NuSeT/blob/master/sample_results/1.png" alt="alt text" width="200">      <img src="https://github.com/yanglf1121/NuSeT/blob/master/sample_results/2.png" alt="alt text" width="200">
<img src="https://github.com/yanglf1121/NuSeT/blob/master/sample_results/3.png" alt="alt text" width="200">      <img src="https://github.com/yanglf1121/NuSeT/blob/master/sample_results/4.png" alt="alt text" width="200">
<img src="https://github.com/yanglf1121/NuSeT/blob/master/sample_results/5.png" alt="alt text" width="300">
</p>

## Reference and citation
This work is inspired from https://github.com/tryolabs/luminoth/tree/master/luminoth/models/fasterrcnn, https://github.com/endernewton/tf-faster-rcnn, https://www.kaggle.com/c/data-science-bowl-2018, https://github.com/matterport/Mask_RCNN.

If you like our NuSeT, here is the paper for this work: https://www.biorxiv.org/content/10.1101/749754v1.
Please cite this paper if NuSeT helps your work.


