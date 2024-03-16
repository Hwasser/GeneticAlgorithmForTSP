import pandas as pd
import random
import threading
import psutil
import time
from library import *
from math import sqrt
from math import floor
import sys

# Author: Victor Hwasser
# Date: 16 october 2022
# Brief: A solution for a Genetic Algorithm

# --------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------

# The genetic algorithm
#
# PARAM data: Graph of nodes to use
# PARAM n_generations: How many generations to go trough before returning the resut
# PARAM population_size: The n om genomes
# PARAM crossover_rate: The rate in range [0,1] on how often to do crossover instead of mutation
# PARAM muration_rate: How many genes in a genome to mutate when mutating - range [0,1]
# PARAM sel_rate: Rate of how many genomes to select from - exact number depends on algorithm
#       in range [0,1]
# PARAM which_mutation: Which mutation algorithm to use - random or scramble route - in range [0,1]
# PARAM which_selection: Which selection method to use, tournament or roulette - in range [0,1]
# PARAM new_blood_rate: How many percent of the worst performing genomes to swap out for newly
#       generated genomes in case a population doesn't improve after one generation, range [0,1]
# PARAM n-best: How many of the elites to keep
# PARAM mutation_swap ("mutation_swap_nodes" or "mtiation_swap edges")
# PARAM crossover ("crossover_1p" or "crossover_2p")
def ga( data,
        n_generations,
        population_size,
        crossover_rate,
        mutation_rate,
        sel_rate,
        which_mutation,
        which_selection,
        new_blood_rate,
        n_best ):
    n_vehicles = 9
    capacity   = 100
    give_up_n = 7 # How many tries to improve each child
    # There are two mutation swap algorithms:

    timer = time.time() # Set a timer
    max_time = 200 # Max 200 seconds runtime

    cur_population = create_population(data, population_size, n_vehicles, capacity)

    for i in range(0, n_generations):
        # Get the best genomes at the top
        cur_population = sort_population(data, cur_population, n_vehicles)
        # Sort the population to get the most fit one first
        best_of_generation = fitness_function(data, n_vehicles, cur_population[0])
        # Infuse some new blood if the generation doesn't get better
        if i > 0 and new_blood_rate > 0:
            if best_old_generation == best_of_generation:
                new_blood = create_population(data, floor(population_size * new_blood_rate), n_vehicles, capacity)
                # improve new blood
                cur_population = cur_population[:(population_size - floor(population_size * new_blood_rate))] + new_blood
        # Keep the n best of this generation in next generation
        new_population = cur_population[0:n_best]
        # Create new population
        for j in range(n_best, population_size):
            child = []
            # re-mutate or re-cross until child is better than parent
            new_is_worse = True
            tries = 0
            while(new_is_worse and tries < give_up_n):
                # Choose selection method
                choose_selection = random.random()
                if choose_selection < which_selection:
                    parent1 = selection_method_roulette(data, n_vehicles, cur_population, sel_rate)
                else:
                    parent1 = selection_method_tournament(data, n_vehicles, cur_population, sel_rate)
                # If cross over or mutation should be used
                if random.random() < crossover_rate:
                    # Choose selection method
                    if choose_selection < which_selection:
                        parent2 = selection_method_roulette(data, n_vehicles, cur_population, sel_rate)
                        child = crossover(parent1, parent2)
                    else:
                        parent2 = selection_method_tournament(data, n_vehicles, cur_population, sel_rate)
                        child = crossover(parent1, parent2)
                else:
                    # Choose between two different mutation methods
                    choose_mutation = random.random()
                    if choose_mutation < which_mutation:
                        child = mutation_swap(parent1, n_vehicles, mutation_rate)
                    else:
                        child = mutation_scrample_route(parent1, n_vehicles, mutation_rate)
                # If the child is actually worse then the parent, run again
                if fitness_function(data, n_vehicles, child) < fitness_function(data, n_vehicles, parent1):
                    new_is_worse = False
                tries += 1

            new_population.append(child)
        cur_population = new_population
        best_old_generation = best_of_generation

        # If final time is reached, return results before all generation is reached
        if time.time() - timer > max_time:
            print("Time out before last generation")
            break

    best = get_best(data, n_vehicles, cur_population)
    print_routes(data, n_vehicles,cur_population)
    print("BEST RESULT:", best)
    print(f"CPU USAGE: {psutil.cpu_percent()}%")
    return best

# --------------------------------------------------------------
# Execution
# ---------------------------------------------------------------

# Multi-threaded runtime of the algorithm
def run_multithreaded(file_name, n_rounds, n_threads, args):
    
    threads = []
    for i in range(0,n_threads):
        # Local copy of data for each thread
        all_nodes = pd.read_csv(file_name).values
        input_vals =  [all_nodes] + args

        # Execute thread
        x = threading.Thread(target=execution, args=args)
        x.start()
        threads.append(x)
        
        # Slightly modify ratios in args for next thread
        args = modify_args(args)
        input_vals =  [all_nodes] + args
    # Joining threads
    for i in range(0,n_threads):
        threads[i].join()

        
# Modifies all ratio values slightly at random for each thread
def modify_args(args):
    new_args = [x for x in args]
    new_args[2] = clamp(new_args[2] - 0.1 + random.random * 0.2)
    new_args[3] = clamp(new_args[2] - 0.1 + random.random * 0.2)
    new_args[4] = clamp(new_args[2] - 0.1 + random.random * 0.2)
    new_args[5] = clamp(new_args[2] - 0.1 + random.random * 0.2)
    new_args[6] = clamp(new_args[2] - 0.1 + random.random * 0.2)
    new_args[7] = clamp(new_args[2] - 0.1 + random.random * 0.2)
    return new_args


# Clamp a float to a ration [0-1]
def clamp(n):
    min_value = 0
    max_value = 1
    
    if n < min_value: 
        return min_value
    elif n > max_value: 
        return max_value
    else: 
        return n 


# Execute the GA for n_rounds with args
def execution(all_nodes, n_rounds, args):
    input_vals = tuple([all_nodes] + args)
    
    results = []
    for i in range(0,n_rounds):
        result = ga(input_vals)
        results.append(result)

        
def main():
    file_name = sys.argv[1]
    n_rounds  = sys.argv[2]
    n_threads = sys.argv[3]
    all_nodes = pd.read_csv(file_name).values

    # NOTE: Values to experiment and play around with: 
    # - n_generations: How many generations to go trough before returning the resut
    # - population_size: The n om genomes
    # - crossover_rate: The rate in range [0,1] on how often to do crossover instead of mutation
    # - mutation_rate: How many genes in a genome to mutate when mutating - range [0,1]
    # - sel_rate: Rate of how many genomes to select from - exact number depends on algorithm
    #      in range [0,1]
    # - which_mutation: Ratio of mutation algorithm to use - random (0) or scramble route (1)
    # - which_selection: Ratio of selection method to use, tournament (0) or roulette (1)
    # - new_blood_rate: Ratio of how many of the worst performing genomes to swap out for
    #     generated genomes in case a population doesn't improve after one generation
    # - n-best: How many of the elites to keep
    # - mutation swap function ("mutation_swap_nodes" or "mtiation_swap edges")
    # - crossover function ("crossover_1p" or "crossover_2p")
    args = [1500, 110, 0.55, 0.40, 0.90, 0.5, 0.96, 0.45, 4, mutation_swap_nodes, crossover_1p]
    
    if n_threads == 1:
        execution(all_nodes, n_rounds, args)
    else:
        run_multithreaded(file_name, n_rounds, n_threads, args)

    

if __name__ == "__main__":
    main()
