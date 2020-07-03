import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import pickle

# number of random runs
num_runs = 50

# parameters of python plot
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 10
plt.rc('axes', labelsize = 14)
plt.rc('xtick', labelsize = 12)   
plt.rc('ytick', labelsize = 12)

# BA, additive, multiplicative and general fitness models
def aggregate_f(growth_model, fitness, degree):
    if growth_model == 1:	    # BA
      return degree
    if growth_model == 2:	    # additive
      return fitness + degree
    if growth_model == 3:	    # multiplicative
      return fitness * degree
    if growth_model == 4:       # general
      return fitness * fitness * degree * degree

# building an initial graph g with two nodes having an edge in between
# give a fitness value using pareto distribution as an attribute to the graph node
def get_init_graph(pareto_param, model, num_iter, num_nodes):

  g = nx.Graph()
  g.add_node(0, fitness = np.random.pareto(pareto_param)) 
  g.add_node(1, fitness = np.random.pareto(pareto_param))
  g.add_edge(0, 1)
  
  fitness_0 = aggregate_f(model, float(g.node[0]['fitness']), g.degree(0))
  fitness_1 = aggregate_f(model, float(g.node[1]['fitness']), g.degree(1))
  
  f_val = []
  f_val.append(float(g.node[0]['fitness']))
  f_val.append(float(g.node[1]['fitness']))

  fitness_arr = []
  fitness_arr.append(fitness_0)
  fitness_arr.append(fitness_1)
  sum_fitness = float(sum(fitness_arr))

  prob_arr = []
  for i in fitness_arr:
    prob_arr.append(i/sum_fitness)
    
  # adding other nodes in the graph
  for i in range (2, num_iter):
    # select a node in the graph to connect the incoming node to
    node_to_connect = np.random.choice(a = range(i), size = num_nodes, p = prob_arr)

    # create the new node with a pareto fitness
    g.add_node(i, fitness = np.random.pareto(pareto_param))
    f_val.append(float(g.node[i]['fitness']))

    # connect with the node selected
    for node in node_to_connect:
      g.add_edge(i, node)
      sum_fitness = sum_fitness - fitness_arr[node] # remove previous fitness value of the node
      fitness_arr[node] = aggregate_f(model, float(g.node[node]['fitness']), g.degree(node)) # get the new fitness value
      sum_fitness = sum_fitness + fitness_arr[node]

    fitness_i = aggregate_f(model, float(g.node[i]['fitness']), g.degree(i)) # fitness of new node added to the graph
    fitness_arr.append(fitness_i)
    sum_fitness = sum_fitness + fitness_arr[i]
    prob_arr = []
    for j in fitness_arr:
      prob_arr.append(j/sum_fitness)
      
  return prob_arr, fitness_arr, sum_fitness, g, f_val

# building network beyond the initial N = 10000 iterations
def get_val(iter_time_change, prob_arr, fitness_arr, num_iter, num_nodes, pareto_param, sum_fitness, g_, model, track, f_val_arr):
  
  fin_vis = np.zeros((iter_time_change))
  
  for run in range (0, num_runs):
    prob = copy.deepcopy(prob_arr)
    fitness_arr_ = copy.deepcopy(fitness_arr)
    sum_fitness_ = sum_fitness
    g = copy.deepcopy(g_)
    f_val_arr_ = copy.deepcopy(f_val_arr)
    visibility_arr = []
    
    for i in range (0, iter_time_change):

      # select a node in the graph to connect the incoming node to
      node_to_connect = np.random.choice(a = range(i + num_iter), size = num_nodes, p = prob)

      # create the new node with a pareto fitness
      if i + num_iter == track - 1:
        index = np.argmax(f_val_arr_)
        g.add_node(i + num_iter, fitness = g.node[index]['fitness'] * 2)  
      else:
        g.add_node(i + num_iter, fitness = np.random.pareto(pareto_param))

      f_val_arr_.append(float(g.node[i]['fitness']))

      # connect with the node selected
      for node in node_to_connect:
        g.add_edge(i + num_iter, node)
        sum_fitness_ = sum_fitness_ - fitness_arr_[node] # remove previous fitness value of the node
        fitness_arr_[node] = aggregate_f(model, float(g.node[node]['fitness']), g.degree(node)) # get the new fitness value
        sum_fitness_ = sum_fitness_ + fitness_arr_[node]

      fitness_i = aggregate_f(model, float(g.node[i + num_iter]['fitness']), g.degree(i + num_iter)) # fitness of new node added to the graph
      fitness_arr_.append(fitness_i)
      sum_fitness_ = sum_fitness_ + fitness_arr_[i + num_iter]
      prob = []

      for j in fitness_arr_:
        prob.append(j/sum_fitness_)

      visibility_arr.append(prob[track - 1])

    fin_vis = fin_vis + visibility_arr
    
  fin_vis = fin_vis / num_runs
  return fin_vis

# getting final result
x = []
for i in range (10000, 100000):
  x.append(i)

def get_results(pareto_param, track, num_iter, num_nodes, iter_time_change):
  
  for model in range (1, 5):
    # build initial graph for a growth model
    prob_arr, fitness_arr, sum_fitness, g_, f_val = get_init_graph(pareto_param, model, num_iter, num_nodes)
    prob_init = copy.deepcopy(prob_arr)
    fitness_init = copy.deepcopy(fitness_arr)
    f_val_init = copy.deepcopy(f_val)
    sum_init = sum_fitness
   
    prob_l = copy.deepcopy(prob_init)
    fitness_l = copy.deepcopy(fitness_init)
    f_val_l = copy.deepcopy(f_val_init)
    vis_arr = get_val(iter_time_change, prob_l, fitness_l, num_iter, num_nodes, pareto_param, sum_init, g_, model, track, f_val_l)
    
    f = str(pareto_param) + str(model) + "10001"
    outfile = open(f, "wb")
    pickle.dump(vis_arr, outfile)
    outfile.close()

    if model == 1:
      label = "BA"
    elif model == 2:
      label = "Add"
    elif model == 3:
      label = "Mult"
    elif model == 4:
      label = "Gen"
      
    plt.plot(x, vis_arr, label = label)
    plt.title("Pareto - " + str(pareto_param) + " / Track node - " + str(track))

pareto = [1, 2, 3] # pareto param
top = [10001] # track 10001th node
num_iter = 10000 # iterations to build the initial graph
num_nodes = 1 # number of nodes to connect to 
iter_time_change = 90000 # iterations to track change in visibility

# show final plots
for i in pareto:
  for j in top:
    plt.figure()
    get_results(i, j, num_iter, num_nodes, iter_time_change)
    plt.legend()
    plt.show()
    plot_title = "Pareto - " + str(i) + " _ Track node - " + str(j)
    plt.savefig(plot_title + str(".png"))