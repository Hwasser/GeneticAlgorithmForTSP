import pandas as pd
import random
import threading
import psutil
import time
from math import sqrt
from math import floor

all_nodes = pd.read_csv("data.csv").values
MAX_COST = 2**30

# Author: Victor Hwasser
# Date: 16 october 2022
# Brief: A solution for a Genetic Algorithm


# --------------------------------------------------------------
# Methods for creating genomes and populations
# ---------------------------------------------------------------

# Cretes a genome. A genome consists of a list of float-points numbers
# where the index is the node taken, integer is in which vehicle
# and mantissa is in what order in ascending order
#
# PARAM data: All nodes in the graph
# PARAM n_vehicles: The number of vehicles to use
# PARAM capactiy: The max capacity of each vehicle
# PARAM total_weight: Total demand / weight of all nodes
# PARAM longest: The longest distance between any node in the graph
#
# Returns: A list with list, each list is a vehicle containg the node it is going to visit
def create_genome(data, n_vehicles, capacity, total_weight, longest):
    # Make a list with available nodes to pickk fram
    nodes_left = list(range(1, len(data)))
    # Fill up every vehicle with random picked packages
    vehicles = []
    for i in range(0, n_vehicles):
        vehicle = fill_one_vehicle(data, nodes_left, n_vehicles, capacity, total_weight, longest)
        vehicles.append(vehicle)

    # If there are nodes left, try to fix!
    if nodes_left != []:
        i = 0
        has_looped = 0 # To see if the algorithm has looped
        # Try to loop through every vehicle to see if there is space
        while nodes_left != []:
            if (vehicles[i][0] > data[nodes_left[0]][1]):
                vehicles[i][1].append(nodes_left.pop())
                has_looped = 0 # Start from new loop
            i = (i + 1) % n_vehicles
            has_looped += 1
            # If solution doesnt work - make a tail recursive call!
            if has_looped > n_vehicles:
                return create_genome(data, n_vehicles, capacity, total_weight, longest)

    return vehicles


# Sub-function for the create genome-function.
# Fills up a vehicle when creating a Gnome by trying to fit nodes from the
# "nodes_left"-list.
#
# PARAM data: All nodes in the graph
# PARAM nodes_left: A list with all nodes left to choose from
# PARAM n_vehicles: The number of vehicles to use
# PARAM capactiy: The max capacity of each vehicle
# PARAM total_weight: Total demand / weight of all nodes
# PARAM longest: The longest distance between any node in the graph
#
# Returns: A list with nodes, representing the route within a vehicle
def fill_one_vehicle(data, nodes_left, n_vehicles, capacity, total_weight, longest):
    # Calculate max weight this vehicle can pick up
    # Only pick up most of the capacity, to avoid next car having too few nodes to chose from
    # Fill up later with round robin
    weight = total_weight / n_vehicles + random.random() * (capacity - total_weight / n_vehicles)
    # Restraint value
    restraint_v = 0.666 # Restraint the distance between nodes in a route
    picked_nodes = [0] # Include dummy variable for checking distance
    i = 1
    while nodes_left != []:
        choice = random.choice(nodes_left)
        weight -= data[choice][1]
        if weight >= 0:
            # RESTRAINT: Make sure new node is not across half the freaking graph from prev nodes
            within_range = True
            for j in range(0, i):
                distance = get_distance(data[picked_nodes[j]], data[choice])
                if distance > longest * restraint_v:
                    within_range = False
            if within_range:
                picked_nodes.append(choice)
                nodes_left.remove(choice)
                i += 1
        if weight <= 0:
            weight += data[choice][1]
            break

    return (weight, picked_nodes[1:]) # Return total weight of nodes taken and skip dummy


# Creates a population
#
# PARAM data: All nodes in the graph
# PARAM g_nomes: The number of genomes in a population
# PARAM n_vehicles: The number of vehicles
# PARAM capacity: Max capacity per vehicle
#
# Returns: A list of genomes
def create_population(data, n_genomes, n_vehicles, capacity):
    total_weight = count_total_weight(data) # Get weight of all nodes
    longest = longest_distance(data) # Get the longest distance between all nodes

    population = []
    # Create each genome and assign random number (random routes) to each vehicle
    for i in range(0, n_genomes):
        genome = list(range(1, len(data)))
        # This will return a list of all vehicles
        genome_raw = create_genome(data, n_vehicles, capacity, total_weight, longest)
        # Assign random routes to each vehicle
        for j in range(0, n_vehicles):
            for n in genome_raw[j][1]:
                genome[n-1] = j + random.random()
        population.append(genome)

    return population


# --------------------------------------------------------------
# Fitness methods
# ---------------------------------------------------------------

# For checking the fitness of a genome
#
# PARAM data: Contains a graph with nodes
# PARAM n_vehicles: How many vehicles whos route to measure
# PARAM genome: The genome to test for fitness
#
# Returns: An integer representing the total distance of all vehicles of the genome
def fitness_function(data, n_vehicles, genome):
    routes_per_vehicle = [[] for _ in range(0, n_vehicles)]
    i = 1
    for g in genome:
        belongs_to = floor(g) # Which vehicle this gene belongs to
        routes_per_vehicle[belongs_to].append((g, i))
        i += 1

    distance = 0
    for route in routes_per_vehicle:
        distance += calculate_route(data, route)

    return distance

# Sub-function for the fitness function
# Calculate route cost for a vehicle -
#
# PARAM data: Contains a graph with nodes
# PARAM vehicle: The vehicle whos route to measure
#
# Returns: An integer representing the distance of the route for the vehicle
def calculate_route(data, vehicle):
    # This means a vehicle has mutated to an empty vehicle to occur!
    if vehicle == []:
        return MAX_COST

    base_node = data[0] # The base node that all paths start from
    # Sort nodes after generated numbers
    vehicle.sort()
    # Distance between base and first node in route
    first_node = get_node(vehicle[0],data) # first node on path
    distance = get_distance(base_node, first_node)
    # Get distance between nodes in the path that the genode has generated
    for i in range(0, len(vehicle)-1):
        cur_node  = get_node(vehicle[i],data)
        next_node = get_node(vehicle[i+1],data)
        distance += get_distance(cur_node, next_node)
    # Add distance between between last node in route and base node
    last_node = get_node(vehicle[-1], data) # last node on path
    distance += get_distance(last_node, base_node)

    return distance

# --------------------------------------------------------------
# Mutation methods
# ---------------------------------------------------------------


# A mutation algorithm, replaces random genes with a random node and route priority
#
# PARAM genome: The gnome to mutate
# PARAM n_vehicles: Number of vehicles in the genome
# PARAM mutation_rate: How many percent of the genome to mutate - range: [0,1]
#
# Returns: A mutated genome
def mutation_swap_nodes(genome, n_vehicles, mutation_rate):
    child = [x for x in genome]
    # Number of mutations we are going to make
    n_mutations = floor(mutation_rate * len(child))
    # Change n_mutations the amount of genes in the genome
    for i in range(0, n_mutations):
        # Take a gene randomly
        gene_simmons = random.randint(0, n_mutations - 1)
        # Re-assemble it to a new vehicle and a new random number
        child[gene_simmons] = random.randint(0, n_vehicles-1) + random.random()

    return child

# A mutation algorithm, swapping random edges, that is - connected nodes
# #
# PARAM genome: The gnome to mutate
# PARAM n_vehicles: Number of vehicles in the genome
# PARAM mutation_rate: How many percent of the genome to mutate - range: [0,1]
#
# Returns: A mutated genome

def mutation_swap_edges(genome, n_vehicle, mutation_rate):
    k = floor(n_vehicle * mutation_rate)
    # Get nodes of all vehicles
    nodes_list = [(genome[i],i+1) for i in range(0, len(genome))]
    child = [x for x in genome] # Copy the genome list
    nodes_list.sort()
    # Switch places on edges
    for _ in range(0, floor(mutation_rate * len(genome))):
        # Get edge1
        edge1 = []
        while edge1 == []:
            random_node = random.randint(0, len(genome)-2)
            # Make sure the nodes are actually connected
            if floor(nodes_list[random_node][0]) == floor(nodes_list[random_node+1][0]):
                edge1 = [nodes_list[random_node][1], nodes_list[random_node+1][1]]
        # Get edge2
        edge2 = []
        while edge2 == []:
            random_node = random.randint(0, len(genome)-2)
            # Make sure the nodes are actually connected
            if floor(nodes_list[random_node][0]) == floor(nodes_list[random_node+1][0]):
                edge2 = [nodes_list[random_node][1], nodes_list[random_node+1][1]]

        child[edge1[0]-1] = child[edge2[0]-1]
        child[edge1[1]-1] = child[edge2[1]-1]
        child[edge2[0]-1] = child[edge1[0]-1]
        child[edge2[1]-1] = child[edge1[1]-1]

    return child


# Scramble the route between nodes within k random vehicles
#
# PARAM genome: The gnome to mutate
# PARAM n_vehicles: Number of vehicles in the genome
# PARAM mutation_rate: How many percent of the genome to mutate - range: [0,1]
#
# Returns: A mutated genome
def mutation_scrample_route(genome, n_vehicle, mutation_rate):
    # How many vehicle routes to scramble
    k = floor(n_vehicle * mutation_rate)
    if k == 0: # Assert that we always at least scamble one
        k = 1
    # Get nodes of all vehicles
    vehicles = [[] for _ in range(0, n_vehicle)]
    for i in range(0, len(genome)):
        v = floor(genome[i])  # vehicle number
        vehicles[v].append((i+1, genome[i]))
    # Scramble routes for k number of cars
    for _ in range(0, k):
        i = random.randint(0, n_vehicle-1)
        vehicles[i] = [(vehicles[i][n][0], i + random.random()) for n in range(0, len(vehicles[i]))]
    # Re-assemble genome
    child = [0 for _ in range(0, len(genome))]
    for v in vehicles:
        for n in v:
            child[n[0]-1] = n[1]
    return child


# --------------------------------------------------------------
# Crossover methods
# ---------------------------------------------------------------

# Crossover single point algorithm
#
# PARAM genome1: Parent genome 1 to cross
# PARAM genome2: Parent genome 2 to cross
#
# Returns: A new genome, representing a "child" of the two parents
def crossover_1p(genome1, genome2):
    # If length of genom1 and genom2 is different something is horribly wrong!
    assert(len(genome1) == len(genome2))
    # Create a random split point
    min_n = 0
    max_n = len(genome1) -1
    split_number = random.randint(min_n,max_n)
    child = []
    for i in range(0, len(genome1)):
        if i < split_number:
            child.append(genome1[i])
        else:
            child.append(genome2[i])

    return child

# Crossover double point algorithm
#
# PARAM genome1: Parent genome 1 to cross
# PARAM genome2: Parent genome 2 to cross
#
# Returns: A new genome, representing a "child" of the two parents
def crossover_2p(genome1, genome2):
    # If length of genom1 and genom2 is different something is horribly wrong!
    assert(len(genome1) == len(genome2))
    # Create two random split points
    min_n = 0
    max_n = len(genome1) // 2
    max_n2 = len(genome1)
    split_number1 = random.randint(min_n,max_n)
    split_number2 = random.randint(max_n+1,max_n2)
    child = []
    for i in range(0, len(genome1)):
        if i < split_number1:
            child.append(genome1[i])
        else:
            if i < split_number2:
                child.append(genome2[i])
            else:
                child.append(genome1[i])

    return child


# Crossover by a uniform distributed number
#
# PARAM genome1: Parent genome 1 to cross
# PARAM genome2: Parent genome 2 to cross
#
# Returns: A new genome, representing a "child" of the two parents
def crossover_uniform(genome1, genome2):
    child = []
    for i in range(0, len(genome1)):
        n = random.randint(1,2)
        if n == 1:
            child.append(genome1[i])
        else:
            child.append(genome2[i])

    return child


# --------------------------------------------------------------
# Selection methods
# ---------------------------------------------------------------


 # Tournament selection
 #
 # PARAM data: A graph of nodes
 # PARAM n_vehicles: Number of vehicles of each genome
 # PARAM population: The population to select from
 # PARAM k: A modifier in range [0,1] which decide number of genomes to pick from.
 #          number of genomes to pick = 10 * k
 #
 # Returns: A selected genome
def selection_method_tournament(data, n_vehicles, population, k):
    k = floor(10 * k)
    # Pick k genomes from the population at random
    challengers = random.sample(population, k)
    best = MAX_COST
    best_c = []
    # Test which genome is best (lowest) and return it
    for c in challengers:
        result = fitness_function(data, n_vehicles, c)
        if result < best:
            best = result
            best_c = c
    return c

# Roulette selection - Actually something of a mix between rank and roulette selection
#
# PARAM data: A graph of nodes
# PARAM n_vehicles: Number of vehicles of each genome
# PARAM population: The population to select from
# PARAM k: A modifier in range [0,1] which decide how many percent of the population to use
#
# Returns: A selected genome
def selection_method_roulette(data, n_vehicles, population, k):
    max_rank = floor(len(population) * k) # Get maximum rank
    sum_rank = sum([x for x in range(1, max_rank + 1)]) # Get sum of all ranks
    p = random.random()
    cur = 0
    for i in range(1, max_rank+1):
        cur += i / sum_rank
        if p <= cur:
            assert population[max_rank-i] != [] # This should not happen
            return population[max_rank-i]

# --------------------------------------------------------------
# Helper functions for main algorithm
# ---------------------------------------------------------------


# Get the best (lowest distance) node of the population
def get_best(data, n_vehicles, population):
    best = MAX_COST
    for genome in population:
        result = fitness_function(data, n_vehicles, genome)
        if result < best:
            best = result
    return best

# Print the routes of all vehicles
def print_routes(data, n_vehicles, population):
    best = MAX_COST
    best_genome = []
    for genome in population:
        result = fitness_function(data, n_vehicles, genome)
        if result < best:
            best = result
            best_genome = genome
    # Get nodes of all vehicles
    vehicles = [[] for _ in range(0, n_vehicles)]
    for i in range(0, len(best_genome)):
        v = floor(best_genome[i])  # vehicle number
        vehicles[v].append((best_genome[i],i+1))
    # Get the route of all vehicles
    for v in range(0,n_vehicles):
        vehicles[v].sort()
        vehicles[v] = [x[1] for x in vehicles[v]]
    print(vehicles)

# Sort population in descending order in relation to its fitness, that is putting
# the most fit in the front for keeping the elite and saving computation for selection
def sort_population(data, population, n_vehicles):
    sorted_pop = []
    # Calculate fitness and sort
    for genome in population:
        score = fitness_function(data, n_vehicles, genome)
        sorted_pop.append((score, genome))
    sorted_pop.sort(reverse=False)
    # Keep the n best
    population = []
    #print("Best of generation is:", sorted_pop[0][0]) # UNCOMMENT FOR TEST OF EACH GEN
    for genome in sorted_pop:
        population.append(genome[1])
    return population

# --------------------------------------------------------------
# Helper functions for other algorithms
# ---------------------------------------------------------------

# Returns the longest distance of all nodes in the graph
def longest_distance(nodes):
    longest = 0
    for m in nodes:
        for n in nodes:
            dist = sqrt((m[2]-n[2])**2 + (m[3]-n[3])**2)
            if dist > longest:
                longest = dist
    return longest

# Distance between node n1 and n2
def get_distance(n1, n2):
    return sqrt( (get_x(n2) - get_x(n1))**2 + (get_y(n2) - get_y(n1))**2 )

# Get x-coordinate of a node
def get_x(n):
    return n[2]

# get y-coordinate of a node
def get_y(n):
    return n[3]

# Get node n from data
def get_node(n, data):
    return data[n[1]]

# Count total weight of all packages / nodes in the graph
def count_total_weight(data):
    weight = 0
    for i in range(0, len(data)):
        weight += data[i][1]
    return weight


# --------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------

# The genetic algorithm
#
# PARAM data: Graph of nodes to use
# PARAM n_generations: How many generations to go trough before returning the resut
# PARAM crossover_rate: The rate in range [0,1] on how often to do crossover instead of mutation
# PARAM muration_rate: How many genes in a genome to mutate when mutating - range [0,1]
# PARAM sel_rate: Rate of how many genomes to select from - exact number depends on algorithm
#       in range [0,1]
# PARAM which_mutation: Which mutation algorithm to use - random or scramble route - in range [0,1]
# PARAM which_selection: Which selection method to use, tournament or roulette - in range [0,1]
# PARAM new_blood_rate: How many percent of the worst performing genomes to swap out for newly
#       generated genomes in case a population doesn't improve after one generation, range [0,1]
# PARAM n-best: How many of the elites to keep
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
    # 1. mutation_swap_nodes
    # 2. mtiation_swap edges
    mutation_swap = mutation_swap_nodes

    # There are two crossover algorithms:
    # 1. crossover_1p - crossover single point
    # 2. crossover_2p - crossover double point
    crossover = crossover_2p

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

#--------------------------------------------------------------------
# TESTING

# Test of the fitness function with optimal values
def test_optimal(all_nodes):
    optima = [3.1, 1.5, 2.2, 0.1, 6.6, 3.3, 0.2, 3.4, 8.4, 6.7,
              1.2, 6.8, 7.5, 1.7, 1.3, 6.3, 1.6, 5.7, 7.3, 0.5,
              2.5, 7.2, 5.1, 5.3, 4.1, 0.7, 7.4, 7.7, 4.3, 7.1,
              0.4, 6.1, 2.4, 2.3, 8.5, 1.1, 2.1, 6.2, 8.2, 6.4,
              4.2, 0.3, 8.6, 5.4, 3.2, 0.6, 8.1, 5.6, 8.3, 5.5,
              1.4, 5.2, 6.5, 7.6]

    #Route #1: 4 7 42 31 20 46 26
    #Route #2: 36 11 15 51 2 17 14
    #Route #3: 37 3 34 33 21
    #Route #4: 1 45 6 8
    #Route #5: 25 41 29
    #Route #6: 23 52 24 44 50 48 18
    #Route #7: 32 38 16 40 53 5 10 12
    #Route #8: 30 22 19 27 13 54 28
    #Route #9: 47 39 49 9 35 43
    print(optima)
    optima_result = fitness_function(all_nodes, 9, optima)
    print(optima_result)

# ----------------------------------------------------
# Runtime

# Multi-threaded runtime of the algorithm
def run_multithreaded(n_threads):
    args = [all_nodes, 1500, 110, 0.55, 0.40, 0.90, 0.5, 0.96, 0.45, 4]
    threads = []
    for i in range(0,n_threads):
        x = threading.Thread(target=ga, args=args)
        threads.append(x)
        x.start()
    # Joining threads
    for i in range(0,n_threads):
        threads[i].join()

# Runtime of the algorithm
results = ga(all_nodes, 1300, 110, 0.55, 0.40, 0.90, 0.5, 0.96, 0.45, 4)
print(results)

# For executing 30 runs
#results = []
#for i in range(0,30):
#    result = ga(all_nodes, 2000, 110, 0.55, 0.40, 0.90, 0.5, 0.96, 0.45, 4)
#    results.append(result)

# RESULTS for previous tested parameters:
# ga(all_nodes, 1000, 100, 0.55, 0.30, 0.65, 0.5, 0.95, 0.20, 5) = 1435
# ga(all_nodes, 1000, 100, 0.55, 0.30, 0.65, 0.5, 0.95, 0.40, 4) = 1509
# ga(all_nodes, 1000, 100, 0.55, 0.30, 0.90, 0.5, 0.95, 0.40, 4) = 1300, 1283
# ga(all_nodes, 1000, 110, 0.55, 0.30, 1, 0.5, 0.95, 0.5, 4) = 1343
# ga(all_nodes, 1000, 110, 0.55, 0.40, 0.90, 0.5, 0.95, 0.4, 4) = 1302
# ga(all_nodes, 1250, 110, 0.55, 0.33, 0.90, 0.5, 0.95, 0.40, 4) = 1234
# ga(all_nodes, 1250, 110, 0.55, 0.33, 0.90, 0.5, 0.95, 0.50, 4) = 1239
# ga(all_nodes, 1250, 110, 0.55, 0.38, 0.90, 0.5, 0.95, 0.4, 4) = 1105
# ga(all_nodes, 1300, 110, 0.55, 0.40, 0.90, 0.5, 0.96, 0.45, 4) = 1276, 1169, 1151, 1212, 1146
