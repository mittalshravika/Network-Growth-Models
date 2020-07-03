import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
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

# BA, additive, multiplicative, general fitness models
def aggregate_f(growth_model, fitness, degree):
    if growth_model == 1:	    # BA
      return degree
    if growth_model == 2:	    # additive
      return fitness + degree
    if growth_model == 3:	    # multiplicative
      return fitness * degree
    if growth_model == 4:     # general
      return fitness * fitness * degree * degree


# building an initial graph g with two nodes having an edge in between
# give a fitness value using pareto distribution as an attribute to the graph node
def get_init_graph(pareto_param, model):

  g = nx.Graph()
  g.add_node(0, fitness = np.random.pareto(pareto_param)) 
  g.add_node(1, fitness = np.random.pareto(pareto_param))
  g.add_edge(0, 1)
  
  fitness_0 = aggregate_f(model, float(g.node[0]['fitness']), g.degree(0))
  fitness_1 = aggregate_f(model, float(g.node[1]['fitness']), g.degree(1))

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
      
  return prob_arr, fitness_arr, sum_fitness, g

"""To get top nodes according to their fitness value"""
def get_top_n(k, f_arr):
  highest_nodes = [i[0] for i in sorted(enumerate(f_arr), key = lambda x:x[1])]
  highest_nodes.reverse()
  top_arr = highest_nodes[0:k]
  return top_arr

# building network beyond the initial N = 10000 iterations
def get_val(iter_time_change, prob_arr, fitness_arr, top, num_iter, num_nodes, pareto_param, sum_fitness, g_, model, iter_):
  
  fin_vis = np.zeros((len(iter_), top))
  
  for run in range (0, num_runs):
    prob = copy.deepcopy(prob_arr)
    fitness_arr_ = copy.deepcopy(fitness_arr)
    sum_fitness_ = sum_fitness
    g = copy.deepcopy(g_)
    vis_arr = []
    
    for i in range (0, iter_time_change):

      # select a node in the graph to connect the incoming node to
      node_to_connect = np.random.choice(a = range(i + num_iter), size = num_nodes, p = prob)

      # create the new node with a pareto fitness
      g.add_node(i + num_iter, fitness = np.random.pareto(pareto_param))

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

      top_node_arr = get_top_n(top, fitness_arr_)
      curr_vis = []
      for j in top_node_arr:
        curr_vis.append(prob[j])
      if (i + num_iter + 1) in iter_:
        vis_arr.append(curr_vis)
    
    fin_vis = fin_vis + vis_arr
    
  fin_vis = fin_vis / num_runs
  return fin_vis

def create_data_for_box_plot(vis, iter_):
  x_ = []
  y_ = []
  for i in range (0, len(vis)):
    arr = vis[i]
    for j in arr:
      y_.append(j)
      x_.append(str(int(iter_[i]/1000)) + "k")
  return x_, y_

iter_ = []
label_ = []
for i in range (10001, 100001):
  if i % 5000 == 0:
    iter_.append(i)
    label_.append(str(int(i/1000)) + "k")

def get_results(pareto_param, top, num_iter, num_nodes, iter_time_change, iter_):
  
  for model in range (1, 5):
    # build initial graph for a growth model
    prob_arr, fitness_arr, sum_fitness, g_ = get_init_graph(pareto_param, model)
    prob_init = copy.deepcopy(prob_arr)
    fitness_init = copy.deepcopy(fitness_arr)
    sum_init = sum_fitness
    
    prob_l = copy.deepcopy(prob_init)
    fitness_l = copy.deepcopy(fitness_init)
    visibility_ = get_val(iter_time_change, prob_l, fitness_l, top, num_iter, num_nodes, pareto_param, sum_init, g_, model, iter_)

    x_, y_ = create_data_for_box_plot(visibility_, iter_)
      
    if model == 1:
      label = "BA"
    elif model == 2:
      label = "Add"
    elif model == 3:
      label = "Mult"
    elif model == 4:
      label = "Gen"
    
    fig = plt.figure(figsize = (12, 6))
    ax = fig.add_subplot(111, label = "1")
    sns.boxplot(x = x_, y = y_, ax = ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Visibility")
    ax.set_xticklabels(label_)
    plot_title = "Pareto Param - " + str(pareto_param) + " _ " + "Top - " + str(top) + " _ " + "Model - " + label
    plt.title(plot_title)
    plt.savefig(plot_title + str(".png"))
    plt.show()

    outfile = open(plot_title, "wb")
    pickle.dump(visibility_, outfile)
    outfile.close()

pareto = [1, 2, 3] # pareto param
top = [50] # vary this accordingly (5, 10 or 50)
num_iter = 10000 # iterations to build the initial graph
num_nodes = 1 # number of nodes to connect to 
iter_time_change = 90000 # iterations to track change in visibility

for i in pareto:
  for j in top:
    get_results(i, j, num_iter, num_nodes, iter_time_change, iter_)