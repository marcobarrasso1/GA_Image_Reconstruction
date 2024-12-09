from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as psns
import matplotlib.pyplot as plt
import imageio
import cv2

target_image = Image.open("../../Downloads/mona_lisa.jpg").convert("RGBA").resize((130, 130))
target_height, target_width = target_image.size

print(target_height, target_width)

def random_color():
    return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

# square side = 5
def add_rectangle(image, size=5, area=25):
    x, y = random.randint(0, target_width-1), random.randint(0, target_height-1)
    rect_width, rect_height = random.randint(1, target_width), random.randint(1, target_height) 
    #rect_width, rect_height = size, size
    #rect_width = random.randint(4, 10)
    rect_height = area / rect_width
    
    image.rectangle([(x,y), (x+rect_width, y+rect_height)], fill=random_color())  
    

def add_constant(ind):
    n_pixels = 50
    
    for i in range(n_pixels):
        x = random.randint(0, ind.shape[0]-1)
        y = random.randint(0, ind.shape[1]-1)
        channel = random.randint(0, 2)  # Random RGB channel
        ind[x, y, channel] = ind[x, y, channel] + random.randint(-10, 10)      

    return ind


def blending(ind1, ind2):
    image1 = Image.fromarray(ind1)
    image2 = Image.fromarray(ind2)
    alpha = random.random()
    return np.array(Image.blend(image1, image2, alpha=alpha))

    
def create_population(population_size):
    initial_population = []
    
    for _ in range(population_size):
        new_image = Image.new("RGBA", (target_height, target_width), color=random_color())
        img = ImageDraw.Draw(new_image)
        n_rectangles = random.randint(3, 6)
        
        for _ in range(n_rectangles):
            add_rectangle(img)
            
        initial_population.append(np.array(new_image))   
        
    return initial_population


def evaluate_fitness(individual):
    return psns(individual, np.array(target_image))


def random_vertical_swap(parent1, parent2):
    """Swap random columns of two images."""
    random_cols= np.random.choice(target_height, 
                                              int(target_height/2), 
                                              replace=False)
    parent1[:, random_cols, :] = parent2[:, random_cols, :]
    
    return parent1


def random_horizontal_swap(parent1, parent2):
    """Swap random columns of two images."""
    random_rows= np.random.choice(target_width, 
                                              int(target_width/2), 
                                              replace=False)
    parent1[random_rows] = parent2[random_rows]
    
    return parent1
    

def get_parent(current_population, current_fitness, tournament_size):
    
    ids = np.random.choice(len(current_population), tournament_size, replace=True)
    candidates_fitness = [current_fitness[i] for i in ids]
    tournament_winner_id = np.argmax(candidates_fitness)
    population_winner_id = ids[tournament_winner_id]
    winner = current_population[population_winner_id]
    
    return winner 


def mutate(ind):
    rand = random.random()
    
    if rand < 0.5:
        ind_image = Image.fromarray(ind)
        _ = ImageDraw.Draw(ind_image)
        
        for i in range(1):
            add_rectangle(_)
            
        return np.array(ind_image)
    else:
        return add_constant(ind)
    
    
def crossover(ind1, ind2):
    rand = random.random()
    
    if rand < 0.3:
        rand2 = random.random()
        if rand2 < 0.5:
            return random_vertical_swap(ind1, ind2)
        else:
            return random_horizontal_swap(ind1, ind2)
    
    else:
        return blending(ind1, ind2)
    
    
def GP():
    population = create_population(50)
    elite_count = 3
    crossover_prob = 0.9
    
    frames = []
    
    for generation in range(20000):
        fitnesses = []
        
        for ind in population:
            fitness = evaluate_fitness(ind)
            fitnesses.append(fitness)
        
        elites_ids = np.argsort(fitnesses)[-elite_count:]
        new_population = []
        tournament_size = max(2, generation // 1000)
        #tournament_size = 5
        
        for i in range(50):
            
            parent1 = get_parent(population, fitnesses, tournament_size=tournament_size)
            parent2 = get_parent(population, fitnesses, tournament_size=tournament_size)
            
            if random.random() < crossover_prob:
                child = crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])
            
            rand = random.random()
            if rand < 0.1:
                child = mutate(child)
                
            new_population.append(child)
            
        for id in elites_ids:
            new_population.append(population[id])
        
        population = new_population
            
        open_cv_image = (population[elites_ids[-1]])
        
        if generation % 100 == 0:
            print(f"Generation: {generation}, avg fitness : {sum(fitnesses) / len(fitnesses)}")
            frames.append(open_cv_image)

    imageio.mimsave("gifs/output_gif.gif", frames, durantion=0.1)
    cv2.imwrite("images/best_image.jpg", open_cv_image) 


    best_ind = np.argmax(fitnesses)
    image = Image.fromarray(population[best_ind])
    plt.imshow(image, cmap="gray")
    plt.axis("off")  # Turn off axes for better visualization
    plt.show()
        

if __name__ == "__main__":
    GP()

