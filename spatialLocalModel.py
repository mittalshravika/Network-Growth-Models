import networkx as nx
import numpy as np
import copy
import math
import random
import pickle

"""Distance between chi params"""
def get_distance(chi_1, chi_2):
  dist = (chi_1[0] - chi_2[0])**2 + (chi_1[1] - chi_2[1])**2
  return math.sqrt(dist)

"""Fitness calculation for local model"""
def aggregate_f(model, chi_1, chi_2, fitness, degree):
  if model == 0:
    return math.exp(-50 * get_distance(chi_1, chi_2)) * fitness * degree

def get_plocal_den(curr_node, nodes, g, model):
  den = 0
  for i in range (0, nodes + 1):
    den = den + aggregate_f(model, g.node[i]['chi_2'], g.node[curr_node]['chi_2'], float(g.node[i]['fitness']), g.degree(i))
  return den

"""Form initial graph"""
def get_init_graph(pareto_param, model):
  
  # building an initial graph g with two nodes having an edge in between
  # give a fitness value using pareto distribution as an attribute to the graph node

  g = nx.Graph()
  g.add_node(0, fitness = np.random.pareto(pareto_param), chi_1 = [0, 0], chi_2 = [random.random(), random.random()]) 
  g.add_node(1, fitness = np.random.pareto(pareto_param), chi_1 = g.node[0]['chi_2'], chi_2 = [random.random(), random.random()])
  g.add_edge(0, 1)
    
  # adding other nodes in the graph
    
  for i in range (2, num_iter):
    
    chi_val = [random.random(), random.random()]
    prob_arr = []
    attach_den = 0
    for node in range (0, i):
      val = aggregate_f(model, g.node[node]['chi_2'], chi_val, g.node[node]['fitness'], g.degree(node))
      prob_arr.append(val)
      attach_den = attach_den + val
    prob_arr = np.asarray(prob_arr)/attach_den
    
    cum_prob_arr = [0]
    for j in range (0, len(prob_arr)):
      cum_prob_arr.append(cum_prob_arr[j] + prob_arr[j])
    
    max_node = 0
    prob_node = random.random()
    for j in range (0, len(cum_prob_arr) - 1):
      if (prob_node >= cum_prob_arr[j] and prob_node < cum_prob_arr[j + 1]):
        max_node = j
        break

    # create the new node with a pareto fitness
    g.add_node(i, fitness = np.random.pareto(pareto_param), chi_1 = g.node[max_node]['chi_2'], chi_2 = chi_val)
    
    # connect with the node selected
    g.add_edge(i, max_node)
  
  fitness_arr = []
  for j in range (0, num_iter):
    if (j % 10000 == 0):
      print (j)
    num = aggregate_f(model, g.node[j]['chi_2'], g.node[j]['chi_2'], float(g.node[j]['fitness']), g.degree(j))
    den = get_plocal_den(j, num_iter - 1, g, model)
    fitness_arr.append(num/den)

  print ("Fitness array - " + str(fitness_arr))
  return fitness_arr, g

# building network beyond the initial N = 10000 iterations
def get_val(iter_time_change, top_nodes, num_iter, pareto_param, g, model):
  
  prob_top_k_nodes = []
  for i in range (0, len(top_nodes)):
    prob_top_k_nodes.append([])
    
  x = []
  for i in range (0, iter_time_change):
    
    chi_val = [random.random(), random.random()]
    prob_arr = []
    attach_den = 0
    for node in range (0, i + num_iter):
      val = aggregate_f(model, g.node[node]['chi_2'], chi_val, g.node[node]['fitness'], g.degree(node))
      prob_arr.append(val)
      attach_den = attach_den + val
    prob_arr = np.asarray(prob_arr)/attach_den
    
    cum_prob_arr = [0]
    for j in range (0, len(prob_arr)):
      cum_prob_arr.append(cum_prob_arr[j] + prob_arr[j])
    
    max_node = 0
    prob_node = random.random()
    for j in range (0, len(cum_prob_arr) - 1):
      if (prob_node >= cum_prob_arr[j] and prob_node < cum_prob_arr[j + 1]):
        max_node = j
        break
    
    # create the new node with a pareto fitness
    g.add_node(i + num_iter, fitness = np.random.pareto(pareto_param), chi_1 = g.node[max_node]['chi_2'], chi_2 = chi_val)

    # connect with the node selected
    g.add_edge(i + num_iter, max_node)
    
    if (i % 500 == 0):
      for j in range (0, len(top_nodes)):
        num = aggregate_f(model, g.node[top_nodes[j]]['chi_2'], g.node[top_nodes[j]]['chi_2'], float(g.node[top_nodes[j]]['fitness']), g.degree(top_nodes[j]))
        den = 0
        for k in range (0, i + num_iter + 1):
          den = den + aggregate_f(model, g.node[k]['chi_2'], g.node[top_nodes[j]]['chi_2'], float(g.node[k]['fitness']), g.degree(k))
        prob_top_k_nodes[j].append(num/den)
    
      x.append(i + num_iter)
      
  return prob_top_k_nodes, x

def get_results(pareto_param, num_iter, iter_time_change):
  
  for model in range (0, 1):
    # build initial graph for a growh model
    fitness_arr, g = get_init_graph(pareto_param, model)
    fitness_init = copy.deepcopy(fitness_arr)
    
    # track changes in node visibility averaged over 50 runs
    top_nodes = []
    for top_k in [1, 5, 10, 30, 50, 100, 200]:
      highest_nodes = [i[0] for i in sorted(enumerate(fitness_init), key = lambda x:x[1])]
      top_node = highest_nodes[len(highest_nodes) - top_k]
      top_nodes.append(top_node)
    
    g_ = copy.deepcopy(g)
    prob, x = get_val(iter_time_change, top_nodes, num_iter, pareto_param, g_, model)

    val = [1, 5, 10, 30, 50, 100, 200] 
    for i in range (0, len(top_nodes)):
      plt.plot(x, prob[i], label = str(val[i]))

pareto = [1, 2, 3]
num_iter = 10000 # iterations to build the initial graph
iter_time_change = 90000 # iterations to track change in visibility

for i in pareto:
  plt.figure()
  get_results(i, num_iter, iter_time_change)
  plt.title("Pareto Param - " + str(i))
  plt.legend()
  plt.savefig("Pareto Param - " + str(i) + " gamma - 50" + str(".png"))
  plt.show()