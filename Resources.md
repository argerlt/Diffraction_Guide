# Resources


## *Python Libraries*

#### [HEXRD: The Highly Extensible X-Ray Diffraction Toolkit](https://github.com/HEXRD/hexrd)
A Suite of X-ray Diffraction Analysis Tools. this is the code used almost exclusively by CHESS for far field diffraction tools. Current best tool in my opinion. Easy access to bleeding edge developments in the field, and large body of dedicated users. However, pooly documented, minimal unit tests, and prone to major breaks. 


#### [ORIX: open source orientation analysis library](https://github.com/pyxem/orix) 
Describes itself as "pythonic MTEX" and is my prefered method for dealing with orientations. Jit-accelerated quaternion-based computations, and does an excellent job of handling convention ambiguity, which is the silent killer of so many projects. Also extremely well documented and open to outside contributions.


## *The original HEXRD Diffraction how-to*
 
#### *[A General Geometric Model for Casting Diffraction](https://hexrd.readthedocs.io/en/latest/_downloads/transforms.pdf)*
Shortened follow-along guide to understanding how HEXRD works.

#### *[Far-field high-energy diffraction microscopy](https://journals.sagepub.com/doi/abs/10.1177/0309324711405761)*
Original full-length paper the guide above was based on

## *Other people's Virtual Diffraction Codes*

[Professor Ashley Bucsek](https://github.com/abucsek/Virtual-Diffraction)
Written in Matlab following the guides above during her PhD. Proven to work for cubic systems. Assumes user is using a single panel GE detector

[Dalton Shadle](https://github.com/daltonshadle/VirtualDiffractometer.git)
PhD student at Cornell

[Professors Paul Dawson and Matt Miller](https://doi.org/10.48550/arXiv.2303.17702)
Recent publication great step-by-step, claims "code is on github", but I cannot find it (published April 3rs, so might not be up yet)

[Diffsims](https://github.com/pyxem/diffsims)
Part of the pyxem team, written primarily by Phillip Crout. I've never used it, and have no idea how well it aligns with the HEXRD methodology

## *Additional Useful Papers*

[On three-dimensional misorientation spaces](https://royalsocietypublishing.org/doi/10.1098/rspa.2017.0274)

[Consistent representations of and conversions between 3D rotations](http://iopscience.iop.org/0965-0393/23/8/083501)

[Connecting heterogeneous single slip to diffraction peak evolution in high-energy monochromatic X-ray experiments](https://doi.org/10.1107/S1600576714005779)





