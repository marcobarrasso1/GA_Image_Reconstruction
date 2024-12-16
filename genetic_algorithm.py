from arg_parse import get_args
from utils import *
import time 
import imageio
import matplotlib.pyplot as plt

#failures: bigger shapes, different fitness function
args = get_args()

target_image = Image.open("target_images/mona_lisa.png").convert("RGB").resize((args.target_height, args.target_width))
print(target_image.size)

population = create_population(args.population_size, target_height=args.target_height, target_width=args.target_width, size=args.size, type=args.shape)
frames = []

start = time.time()
for generation in range(args.generations):
        
    fitnesses = []
    for ind in population:
        fitness = evaluate_fitness(ind, target_image)
        fitnesses.append(fitness)
    
    elites_ids = np.argsort(fitnesses)[-args.n_elites:]
    new_population = []
    tournament_size = max(2, int((generation / args.generations) * 10))

    if generation % 100 == 0:
        best_ind = (population[elites_ids[-1]])
        frames.append(best_ind)
        end = time.time()
        #print(f"Generation: {generation}, avg fitness : {sum(fitnesses) / len(fitnesses)}, time per 100 generations: {(end - start):.2f}")
        print(f"Generation: {generation}, avg fitness : {sum(fitnesses) / len(fitnesses):.2f}")
        mean = sum(fitnesses) / len(fitnesses)
        sd = (sum((ind - mean) ** 2 for ind in fitnesses) / (len(fitnesses) - 1)) ** 0.5
        #print(f"Generation: {generation}, fitness {mean:.3f}, sd {sd:.4f}")
        start = time.time()
    
    for i in range(args.population_size):
        
        parent1 = get_parent(population, fitnesses, tournament_size=tournament_size)
        parent2 = get_parent(population, fitnesses, tournament_size=tournament_size)
        
        if random.random() < args.p_cross:
            child = crossover(parent1, parent2)
        else:
            child = random.choice([parent1, parent2])
        
        rand = random.random()
        if rand < args.p_mut:
            child = mutate(child, size=args.size, type=args.shape)
            
        new_population.append(child)
        
    for id in elites_ids:
        new_population.append(population[id])
    
    population = new_population
        
        

imageio.mimsave(f"gifs/{args.gif}.gif", frames)
plt.imsave(f"output_images/{args.gif}.png", best_ind) 

