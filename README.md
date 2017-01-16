# thoughtvec-imagesearch

Description-based image search based on the paper 'Skip-Thought Vectors' by Kiros et al.

### Release notes

Please note that even though we know this approach works (cf. Kiros et al 2015),
my use and implementation of Kiros' image-sentence ranking model is still a work in progress. 

In particular: 
  1. There are some performance issues caused by interfacing with Kiros' implementation.
  2. Parts of my own code need to be cleaned up.
  3. Parts of Kiros' code and my own could benefit from being jointly refactored to improve performance.
  4. My code appears to provide different performance statistics than Kiros' code.
  5. My implementation of the ConvNet uses a resize-and-pad approach.  
     VGG16 expects a different image shape than the MS COCO images possess.  
     As a result, my implementation may differ from the implementation used by Kiros et al. 2015.  

### Getting started

1. In order for this code to run, you must clone the skip-thoughts repo:  
    https://github.com/ryankiros/skip-thoughts

  1. As per the readme in the skip-thoughts repo, you must download the pretrained skip-thoughts model parameters
    from the utoronto website. See Kiros' readme for details.

  2. In addition, (as per the skip-thoughts readme) you must edit the skipthoughts.py file to use the correct paths to models and data.

2. In addition, to use this code, you must download the weights for VGG16 from here:  
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

3. In addition, to train the model for embedding ConvNet codes and Skip-Thought Vectors into the same space,
   you must download the MS COCO images and annotations from here:  
    http://mscoco.org/dataset/#download

4. Finally, you must edit the first code snippet in thoughtvec-imagesearch.ipynb to point to the correct paths for
  1. the VGG16 weights, 
  2. the directory for the MS COCO data,
  3. the model directory for *your* models (which ideally will not the same dir as for the skipthoughts model directory).


### Reference

If you use this program, please cite these papers in your readme:

    Skip-Thought Vectors
    R. Kiros, Y. Zhu, R. Salakhutdinov, R. Zemel, A. Torralba, R. Urtasun, and S. Fidler
    arXiv:1506.06726

    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556

    

