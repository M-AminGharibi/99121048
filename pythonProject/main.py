import numpy as np
from deap import creator, base, tools, algorithms

# Define the warehouse layout
warehouse_map = np.array([[1, 1, 1, 0, 0],
                          [1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1],
                          [0, 0, 0, 1, 1]])

# Define a function to generate a random individual
def generate_random_individual():
    return [np.random.randint(2) for _ in range(warehouse_map.size)]

# Define a function to calculate fitness (in this case, maximize free space)
def calculate_fitness(individual):
    total_space = np.sum(warehouse_map)
    used_space = np.sum(warehouse_map) - np.sum(individual)
    return total_space - used_space,

# Define the genetic algorithm parameters
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", generate_random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=100)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", calculate_fitness)

# Define the main function to run the genetic algorithm
def main():
    population = toolbox.population()
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=True)
    best_individual = tools.selBest(population, k=1)[0]
    print("Best individual:", best_individual)

if __name__ == "__main__":
    main()
