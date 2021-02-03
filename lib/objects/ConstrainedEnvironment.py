# import random

# import graph_tool.all as gt
# import matplotlib.pyplot as plt
# import numpy as np

# from lib.objects import Environment


# class ConstrainedEnvironment(Environment):
#     def __init__(self):
#         self.__as_graph()
#         self.node_coords = []

#     def __as_graph(self):

#         g = gt.Graph()
#         g.vp["name"] = g.new_vertex_property("string")
#         g.vp["pos"] = g.new_vertex_property("vector<float>")
#         g.vp["cost"] = g.new_vertex_property("double")
#         g.ep["cost"] = g.new_edge_property("double")

#         self.node_map = {}
#         self.graph = g

#     def add_vertex(self, q, cost):
#         node_name = self.get_node_name(q)
#         new_node = self.graph.add_vertex(1)

#         self.node_map[node_name] = new_node
#         self.node_coords.append(q.reshape(-1, 1))
#         self.graph.vp["name"][new_node] = node_name
#         self.graph.vp["pos"][new_node] = q.reshape(-1,).tolist()
#         self.graph.vp["cost"][new_node] = cost

#     def add_edge(self, src_coord, tgt_coord, cost):
#         src = self.get_node(src_coord)
#         tgt = self.get_node(tgt_coord)

#         e = self.graph.add_edge(src, tgt)
#         self.graph.ep["cost"][e] = cost

#     def get_node_name(q):
#         return "({}, {}, {})".format(*q.reshape(-1,))

#     def get_node(self, q):
#         return self.node_map[self.get_node_name(q)]
