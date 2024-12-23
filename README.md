# Genetic Algorithm for Image Reconstruction
## Intro

Genetic algorithm implementation to approximate a target image by composing it with simple geometric shapes such as triangles, circles, rectangles, lines and characters.
It evoleves images by iteratively change the properties of the shapes (how many, position, color, size) to minimize the difference between the reconstructed image and the original target image.

## Requirements

* scikit-image
* imageio
* numpy
* opencv-python
* Pillow
* scikit-image
* matplotlib

## Algorithm details

* Population size: 50
* Tournament selection: size starting from 2 and gradually increases up to 10
* Crossover probability: 0.8
* Mutation probability: 0.2
* Elitism number: 3

  
