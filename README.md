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
### Population
Pupulation size of 50 where an individual is represented as an 256x256x3 RGB image.

### Fitness Function
For measuring image similarity i used the Peak Signal-to-Noise ratio metric. It is calculated as:

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right)
$$

Where:
- **MAX**: Maximum possible pixel value (e.g., 255 for 8-bit images).
- **MSE**: Mean Squared Error, defined as:
- $$
  \text{MSE} = \frac{1}{H \cdot W} \sum_{x=1}^{H} \sum_{y=1}^{W} \left( I_t(x, y) - I_r(x, y) \right)^2
  $$

$$
  \text{MSE} = \frac{1}{H \cdot W} \sum_{x=1}^{H} \sum_{y=1}^{W} \left( I_t(x, y) - I_r(x, y) \right)^2
$$
  
Tournament selection: size starting from 2 and gradually increases up to 10
* Crossover probability: 0.8
* Mutation probability: 0.2
* Elitism number: 3

  
