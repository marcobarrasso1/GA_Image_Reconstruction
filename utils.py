from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio
import cv2
from skimage.metrics import structural_similarity as ssim

def random_color():
    return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

# square side = 5
def add_rectangle(image, target_height, target_width, size):
    x, y = random.randint(0, target_width-1), random.randint(0, target_height-1)
    rect_width, rect_height = random.randint(3, size), random.randint(3, size) 
    
    image.rectangle([(x,y), (x+rect_width, y+rect_height)], fill=random_color())  
    

def add_constant(ind):
    n_pixels = 50
    
    for i in range(n_pixels):
        x = random.randint(0, ind.shape[0]-1)
        y = random.randint(0, ind.shape[1]-1)
        channel = random.randint(0, 3)  # Random RGBA channel
        ind[x, y, channel] = np.clip(
            int(ind[x, y, channel]) + random.randint(-10, 10), 0, 255
        )      

    return ind


def blending(ind1, ind2):
    image1 = Image.fromarray(ind1)
    image2 = Image.fromarray(ind2)
    alpha = random.random()
    return np.array(Image.blend(image1, image2, alpha=alpha))

    
def create_population(population_size, target_height, target_width, size):
    initial_population = []
    
    for _ in range(population_size):
        new_image = Image.new("RGBA", (target_height, target_width), color=random_color())
        img = ImageDraw.Draw(new_image)
        n_rectangles = random.randint(3, 6)
        
        for _ in range(n_rectangles):
            add_rectangle(img, target_height, target_width, size)
            
        initial_population.append(np.array(new_image))   
        
    return initial_population


def evaluate_fitness(individual, target):
    return psnr(individual, np.array(target))
    

def vertical_swap(parent1, parent2):
    random_col= random.randint(0, parent1.shape[1]-1)
    parent1[:, 0:random_col, :] = parent2[:, 0:random_col, :]
    
    return parent1
    
    
def horizontal_swap(parent1, parent2):
    random_row = random.randint(0, parent1.shape[0]-1)
    parent1[0:random_row] = parent2[0:random_row]
    
    return parent1


def random_vertical_swap(parent1, parent2):
    """Swap random columns of two images."""
    random_cols= np.random.choice(parent1.shape[0], 
                                              int(parent1.shape[0]/2), 
                                              replace=False)
    parent1[:, random_cols, :] = parent2[:, random_cols, :]
    
    return parent1


def random_horizontal_swap(parent1, parent2):
    """Swap random columns of two images."""
    random_rows= np.random.choice(parent1.shape[0], 
                                              int(parent1.shape[0]/2), 
                                              replace=False)
    parent1[random_rows] = parent2[random_rows]
    
    return parent1
    

def pixelwise_change(parent1, parent2):
    mask = np.random.choice([True, False], size=parent1.shape)
    child = np.where(mask, parent1, parent2)
    
    return child


def get_parent(current_population, current_fitness, tournament_size):
    
    ids = np.random.choice(len(current_population), tournament_size, replace=True)
    candidates_fitness = [current_fitness[i] for i in ids]
    tournament_winner_id = np.argmax(candidates_fitness)
    population_winner_id = ids[tournament_winner_id]
    winner = current_population[population_winner_id]
    
    return winner 


def mutate(ind, size):
    rand = random.random()
    
    if rand < 0.7:
        ind_image = Image.fromarray(ind)
        _ = ImageDraw.Draw(ind_image)
        
        for i in range(1):
            add_rectangle(_, ind.shape[1], ind.shape[0], size)
            
        return np.array(ind_image)
    else:
        return add_constant(ind)
    
    
def crossover(ind1, ind2):
    rand = random.random()
    
    if rand < 0.3:
        rand2 = random.randint(0, 3)
        if rand2 == 0:
            return random_vertical_swap(ind1, ind2)
        elif rand2 == 1:
            return random_horizontal_swap(ind1, ind2)
        elif rand2 == 2:
            return vertical_swap(ind1, ind2)
        else:
            return horizontal_swap(ind1, ind2)
    
    elif rand < 0.95:  
        return blending(ind1, ind2)
    
    else:  
        return pixelwise_change(ind1, ind2)
    
    

