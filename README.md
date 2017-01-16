# thoughtvec-imagesearch

Description-based image search based on the paper 'Skip-Thought Vectors' by Kiros et al.


### Getting started

1. In order for this code to run, you must clone the skip-thoughts repo:
    https://github.com/ryankiros/skip-thoughts

1a. As per the readme in the skip-thoughts repo, you must download the pretrained skip-thoughts model parameters
    from the utoronto website. See Kiros' readme for details.

1b. In addition, (as per the skip-thoughts readme) you must edit the skipthoughts.py file to use the correct paths to models and data.

2. In addition, to use this code, you must download the weights for VGG16 from here:
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

3. In addition, to train the model for embedding ConvNet codes and Skip-Thought Vectors into the same space,
   you must download the MS COCO images and annotations from here:
    http://mscoco.org/dataset/#download

4. Finally, you must edit the first code snippet in thoughtvec-imagesearch.ipynb to point to the correct paths for
   (i)   the VGG16 weights, 
   (ii)  the directory for the MS COCO data,
   (iii) the model directory for *your* models (which ideally will not the same dir as for the skipthoughts model directory).


### Reference

    If you use this program, please cite these papers in your readme:

    Skip-Thought Vectors
    R. Kiros, Y. Zhu, R. Salakhutdinov, R. Zemel, A. Torralba, R. Urtasun, and S. Fidler
    arXiv:1506.06726

    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556

    

