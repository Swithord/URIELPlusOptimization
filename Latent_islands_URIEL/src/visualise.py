import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Digraph
import os

def graph_islands(islands, feature_names = None, name = None, path = None):
  n_islands = len(islands)
  if name == None:
    name = "untitled graph_islands call"
  if feature_names is None:
    total_features = 0
    for i in range(n_islands):
      total_features += len(islands[i]["indices"])
    feature_names = [f"(feature: {i})" for i in range(total_features)]

  graph = Digraph(format = 'png')
  graph.attr(rankdir='LR')
  # graph.attr(rankdir='TB')
  # top to bottom somehow isn't working
  # need to use subgraphs

  for i, island in enumerate(islands):
    island_latent = f"Z{i}"
    graph.node(island_latent, shape = "circle", label = island_latent)
    for index in island["indices"]:
      feature_node = f"feature {index}"
      graph.node(feature_node, shape = "ellipse", label = feature_names[index])
      graph.edge(island_latent, feature_node)
  # graph.render("latent factor islands", view = True)

  # # make invisible edges for layout purposes
  # for i in range(len(islands) - 1):
  #       graph.edge(f"Z{i}", f"Z{i+1}", style='invis')
  # graph.attr(rankdir='LR')
  if path is None:
    path = os.path.join("outputs", name)
  graph.render(path, cleanup = True, view = False)
  return graph