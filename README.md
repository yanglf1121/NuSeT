# NuSeT
NuSeT: A Deep Learning Tool for Reliably Separating and Analyzing Crowded Cells

## To run the code
The following packages are needed for NuSeT to run:
1. tensorflow (*pip3 install tensorflow / pip3 install tensorflow-gpu*)
2. PIL (*pip3 install Pillow*)
3. numpy (*pip3 install numpy*)
4. scikit-image (*pip3 install scikit-image*)
5. tqdm (*pip3 install tqdm*)

After finishing installing packages, navigate to the root folder of this repo (NuSeT/), in the command line run *python3 NuSeT.py*.

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
![](https://github.com/yanglf1121/NuSeT/blob/master/sample_results/1.png | width=100)






