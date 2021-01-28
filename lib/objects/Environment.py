import copy

import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from . import Obstacle


class Environment:
    def __init__(self, Arena, StaticObstacles=[], DynamicObstacles=[]):
        self.Arena = Arena

        # initialise the configuration spaces
        self.c_free = self.reachable_points([])  # all point in the arena
        self.c_stat = self.reachable_points([])  # all point in the arena
        self.c_dyn = self.reachable_points([])  # all point in the arena

        # Graph representation of the environment
        self.graph = None
        self.node_map = {}
        self.__as_graph()

        # Put this in a getter and setter as to update the static and dynamic obstacle configuration
        # spaces, and update the free graph.
        self.StaticObstacles = StaticObstacles
        self.DynamicObstacles = DynamicObstacles

        # Now we have the graph and free configuration space, we can add our obstacles.
        # self.c_dyn = self.reachable_points(self.DynamicObstacles) # dynamic objects

    @property
    def StaticObstacles(self):
        return self._StaticObstacles

    @StaticObstacles.setter
    def StaticObstacles(self, value):
        """Setter for StaticObstacles. Must be a list, even if there is only one.

        Args:
            value (dict):

        Raises:
            TypeError: Must be a pika.adapters.blocking_connection.BlockingChannel
        """
        if not isinstance(value, list):
            raise TypeError("StaticObstacles must be a list object.")

        for obs in value:
            if not isinstance(obs, Obstacle):
                raise TypeError("StaticObstacles contents must be an Obstacle object.")

        self.c_stat = self.__update_inventory(
            value, copy.deepcopy(self.c_stat), "stat_obj"
        )
        self.c_reach = self.current_reachable()
        self._StaticObstacles = value

    @property
    def DynamicObstacles(self):
        return self._DynamicObstacles

    @DynamicObstacles.setter
    def DynamicObstacles(self, value):
        """Setter for DynamicObstacles. Must be a list, even if there is only one.

        Args:
            value (dict):

        Raises:
            TypeError: Must be a pika.adapters.blocking_connection.BlockingChannel
        """
        if not isinstance(value, list):
            raise TypeError("DynamicObstacles must be a list object.")

        for obs in value:
            if not isinstance(obs, Obstacle):
                raise TypeError("DynamicObstacles contents must be an Obstacle object.")

        self.c_dyn = self.__update_inventory(
            value, copy.deepcopy(self.c_dyn), "dyn_obj"
        )
        self.c_reach = self.current_reachable()
        self._DynamicObstacles = value

    def __update_edge_vertex_status(self, row_idx, col_idx, obj_type):

        for row, col in zip(row_idx, col_idx):
            node_loc = (self.Arena.x_mesh[row, col], self.Arena.y_mesh[row, col])
            node_name = "({}, {})".format(*node_loc)
            node = self.node_map[node_name]
            self.graph.vp[obj_type][node] = not self.graph.vp[obj_type][node]
            edges = self.graph.get_out_edges(node)
            for e in edges:
                e = self.graph.edge(*e)
                self.graph.ep["active"][e] = not self.graph.ep["active"][e]
                if self.graph.ep["active"][e]:
                    self.graph.ep["weight"][e] = self.graph.ep["dist"][e]
                else:
                    self.graph.ep["weight"][e] = np.inf

    def __update_inventory(self, obj_list, prev_obj, obj_type):
        # We can now update a few internal parameters.
        c_obj = self.reachable_points(obj_list)

        # Get ther difference.
        delta_obj = c_obj - prev_obj

        # Make vertex a static object
        row_idx, col_idx = np.where(delta_obj > 0)
        self.__update_edge_vertex_status(row_idx, col_idx, obj_type)

        # Unmake vertex a static object, i.e, reachable space
        row_idx, col_idx = np.where(delta_obj < 0)
        self.__update_edge_vertex_status(row_idx, col_idx, obj_type)

        return c_obj

    def __as_graph(self):
        """Build the base graph of the configuration space.

        This deals with ALL connections in the free configuration space.

        This is a slow process, but only need to be done once.

        """
        g = gt.Graph()

        # Add some graph properties
        g.vp["name"] = g.new_vertex_property("string")
        g.vp["pos"] = g.new_vertex_property("vector<float>")
        g.vp["stat_obj"] = g.new_vertex_property("bool")
        g.vp["dyn_obj"] = g.new_vertex_property("bool")
        g.ep["dist"] = g.new_edge_property("double")
        g.ep["weight"] = g.new_edge_property("double")
        g.ep["active"] = g.new_edge_property("bool")

        # Add the vertices
        for i, row in enumerate(self.c_free):
            for j, col in enumerate(row):
                node_loc = (self.Arena.x_mesh[i, j], self.Arena.y_mesh[i, j])
                node_name = "({}, {})".format(*node_loc)

                if node_name in self.node_map:
                    continue

                new_node = g.add_vertex(1)
                self.node_map[node_name] = new_node
                g.vp["name"][new_node] = node_name
                g.vp["pos"][new_node] = node_loc

        # Now add the edges
        for i, row in enumerate(self.c_free):
            for j, col in enumerate(row):

                # reachable points have a value of 0. This is if it
                # reachable, it will have a value.

                if self.c_free[i, j]:
                    continue

                src_loc = (self.Arena.x_mesh[i, j], self.Arena.y_mesh[i, j])
                src_name = "({}, {})".format(*src_loc)
                src = self.node_map[src_name]

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if (not di) and (not dj):
                            continue

                        # Again, if the adjacent point has a value, it is in invalid traversal.
                        if self.c_free[i + di, j + dj]:
                            continue

                        tgt_loc = (
                            self.Arena.x_mesh[i + di, j + dj],
                            self.Arena.y_mesh[i + di, j + dj],
                        )
                        tgt_name = "({}, {})".format(*tgt_loc)
                        tgt = self.node_map[tgt_name]

                        e = g.add_edge(src, tgt)

                        q_src = np.array(g.vp["pos"][src])
                        q_tgt = np.array(g.vp["pos"][tgt])
                        q_dst = q_src - q_tgt
                        q_dst = np.linalg.norm(q_dst)
                        g.ep["dist"][e] = q_dst
                        g.ep["weight"][e] = q_dst
                        g.ep["active"][e] = True

        self.graph = g

    def current_reachable(self):
        """Get the reachable points in the configuration space, i.e., element-wise max
        of the the free configuration space, and the constrained configuration space
        due to static or moving obstacles.
        """
        return np.maximum(self.c_dyn, self.c_stat)

    def all_obstacles(self):
        return self.StaticObstacles + self.DynamicObstacles

    def reachable_points(self, obstacle_list, epsilon=1, max_val=100):
        """Get the reachble array based on the obstacle_list provided.

        A zero indicates the cell is reachable, and max_val indicates the cell is not reachable.

        """
        p_reach = np.zeros(self.Arena.x_mesh.ravel().shape)
        Q = np.block([[self.Arena.x_mesh.ravel()], [self.Arena.y_mesh.ravel()]])

        for o in obstacle_list:
            o_vec = o.as_vec()
            q_obs = o_vec[:2]
            r_obs = o_vec[2, 0]

            dq = Q - o.as_vec()[:2, :]
            d = np.linalg.norm(dq, axis=0)
            d = np.where(d <= (r_obs + epsilon), max_val, 1)
            o_idx = np.where(d == max_val)[0].tolist()
            for idx in o_idx:
                p_reach[idx] = max_val

        # Reshape
        p_reach = p_reach.reshape(self.Arena.x_mesh.shape)

        # Set the boundary elements
        p_reach[0, :] = max_val
        p_reach[-1, :] = max_val
        p_reach[:, 0] = max_val
        p_reach[:, -1] = max_val

        return p_reach

    def reachable_coordinates(self):

        q_reach = np.array(
            [
                (self.Arena.x_mesh[i, j], self.Arena.y_mesh[i, j])
                for i, row in enumerate(self.c_reach)
                for j, col in enumerate(row)
                if not col
            ]
        )

        return q_reach

    def reconstruct_path(self, src, tgt, preds):
        """Reconstruct a path back form the target
        to the source using the predecessor array that
        is returned from the dijkstra or astar searches.

        Args:

        Returns:

        """

        vertices = []
        while int(tgt) != int(src):
            vertices.append(int(tgt))
            tgt = preds.a[int(tgt)]

        vertices.reverse()

        return vertices

    def get_route(self, src, tgt):

        src_node = self.graph.vertex(src)
        tgt_node = self.graph.vertex(tgt)
        dists = self.graph.ep["weight"]

        dist_map, pred_map = gt.astar_search(self.graph, source=src_node, weight=dists)
        min_dist = np.inf

        node_order = [int(src)]
        if dist_map[tgt] < min_dist:
            node_order.extend(self.reconstruct_path(src_node, tgt_node, pred_map))

        return node_order

    def plot_environment(self, ax, nodes_visited, debug=False):
        ax.clear()
        for o in self.all_obstacles():
            o = o.as_vec()
            circle = plt.Circle(o[:2], o[2])
            ax.plot(o[0], o[1], "w+")
            ax.add_artist(circle)
        ax.contour(self.Arena.x_mesh, self.Arena.y_mesh, self.c_reach)

        path = []
        for v in nodes_visited:
            vert = self.graph.vertex(v)
            path.append(self.graph.vp["pos"][vert])
        ax.plot(np.array(path)[:, 0], np.array(path)[:, 1], "r--", alpha=0.5)

        if debug:
            for node in self.graph.vertices():
                if self.graph.vp["stat_obj"][node]:
                    ax.plot(*self.graph.vp["pos"][node], "b+")
                if self.graph.vp["dyn_obj"][node]:
                    ax.plot(*self.graph.vp["pos"][node], "g+")

            for edge in self.graph.edges():
                edge = self.graph.edge(*edge)
                if self.graph.ep["active"][edge]:
                    ax.plot(*self.graph.vp["pos"][edge.source()], "k.", alpha=0.1)
                    ax.plot(*self.graph.vp["pos"][edge.target()], "k.", alpha=0.1)

        ax.grid()
        ax.axis("equal")

        return ax
