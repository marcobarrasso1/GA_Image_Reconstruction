from arg_parse import get_args
from utils import *
import time 

args = get_args()

target_image = Image.open("images/mona_lisa.jpg").convert("RGBA").resize((args.target_height, args.target_width))
print(target_image.size)

population = create_population(args.population_size, target_height=args.target_height, target_width=args.target_width, size=args.size, type=args.shape)
frames = []

start = time.time()
for generation in range(args.generation):
        
    fitnesses = []
    for ind in population:
        fitness = evaluate_fitness(ind, target_image)
        fitnesses.append(fitness)
    
    elites_ids = np.argsort(fitnesses)[-args.n_elites:]
    new_population = []
    tournament_size = max(2, generation // 1000)
    #tournament_size = 5
    
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
        
    best_ind = (population[elites_ids[-1]])
    
    if generation % 100 == 0:
        frames.append(best_ind)
        end = time.time()
        print(f"Generation: {generation}, avg fitness : {sum(fitnesses) / len(fitnesses)}, time per 100 generations: {(end - start):.2f}")
        start = time.time()

imageio.mimsave("gifs/output_gif_letter10.gif", frames)
plt.imsave("images/best_image.jpg", best_ind) 

