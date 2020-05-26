# Image Registration: Large Deform Diffeomorphism metric mapping
![](https://img.shields.io/badge/<Implementation>-<lddmm>-<success>)

[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R5R11K2H4)


## Getting Started

Simply open the jupyter notebook and see how some demo on pictures that we uploaded with this repository

### Prerequisites

What things you need to install the software and how to install them

```
scikit-image
matplotlib
numpy
pytorch
notebook
scipy
```

### Installing

Here are the steps to follow

```
conda install environment.yml
conda activate lddmm
```

## Running the tests

Simply run main.py

```
python3 main.py
```

### Break down into end to end tests

Our tests are working on three different sets of images. Here are some examples of results and their explanation.

#### Population representatating template
- Random initialization of control points
- momentum of 0
- two steps gradient descent momentum and control points
- loss: MSE
- regularization: euclidian norm on momentum (integrated on path)


Starting from mean image initialization
![](https://github.com/miki998/Unsupervised-Image-normalization-Atlas/blob/master/example1.png)


After 80 iterations
![](https://github.com/miki998/Unsupervised-Image-normalization-Atlas/blob/master/example2.png)

### Normalization of input to template
- 
- 
- 
!["NOT YET"]

## Deployment

None yet, you can do some pull requests to me

## Built With

* [python3](https://www.python.org/download/releases/3.0/) - The web framework used

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors
Michael Chan
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments
...








