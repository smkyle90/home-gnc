import copy

import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .Obstacle import Obstacle


class Environment:
    def __init__(self, arena, static_obs=[], dynamic_obs=[]):
        self.arena = arena

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
        self.static_obs = static_obs
        self.dynamic_obs = dynamic_obs

        # Now we have the graph and free configuration space, we can add our obstacles.
        # self.c_dyn = self.reachable_points(self.dynamic_obs) # dynamic objects

    @property
    def static_obs(self):
        return self._static_obs

    @static_obs.setter
    def static_obs(self, value):
        """Setter for static_obs. Must be a list, even if there is only one.

        Args:
            value (dict):

        Raises:
            TypeError: Must be a pika.adapters.blocking_connection.BlockingChannel
        """
        if not isinstance(value, list):
            raise TypeError("static_obs must be a list object.")

        for obs in value:
            if not isinstance(obs, Obstacle):
                raise TypeError("static_obs contents must be an Obstacle object.")

        self.c_stat = self.__update_inventory(
            value, copy.deepcopy(self.c_stat), "stat_obj"
        )
        self.c_reach = self.current_reachable()
        self._static_obs = value

    @property
    def dynamic_obs(self):
        return self._dynamic_obs

    @dynamic_obs.setter
    def dynamic_obs(self, value):
        """Setter for dynamic_obs. Must be a list, even if there is only one.

        Args:
            value (dict):

        Raises:
            TypeError: Must be a pika.adapters.blocking_connection.BlockingChannel
        """
        if not isinstance(value, list):
            raise TypeError("dynamic_obs must be a list object.")

        for obs in value:
            if not isinstance(obs, Obstacle):
                raise TypeError("dynamic_obs contents must be an Obstacle object.")

        self.c_dyn = self.__update_inventory(
            value, copy.deepcopy(self.c_dyn), "dyn_obj"
        )
        self.c_reach = self.current_reachable()
        self._dynamic_obs = value

    def nearest_node(self, x, y):
        q_sample = np.array([x, y]).reshape(-1, 1)
        q_nodes = self.reachable_coordinates().T
        node_delta = q_nodes - q_sample

        min_idx = np.argmin(np.linalg.norm(node_delta, axis=0))

        return tuple(q_nodes[:, min_idx])

    def get_node_name(self, x, y):
        return "({}, {})".format(x, y)

    def get_node_by_coord(self, x, y):
        """Get the nearest node by location
        """
        node_name = self.get_node_name(x, y)

        if node_name not in self.node_map:
            node_loc = self.nearest_node(x, y)
            node_name = self.get_node_name(*node_loc)

        return self.node_map[node_name]

    def __update_edge_vertex_status(self, row_idx, col_idx, obj_type):

        for row, col in zip(row_idx, col_idx):
            node_loc = (self.arena.x_mesh[row, col], self.arena.y_mesh[row, col])
            node = self.get_node_by_coord(*node_loc)
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
                node_loc = (self.arena.x_mesh[i, j], self.arena.y_mesh[i, j])
                node_name = self.get_node_name(*node_loc)

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

                src_loc = (self.arena.x_mesh[i, j], self.arena.y_mesh[i, j])
                src = self.get_node_by_coord(*src_loc)

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if (not di) and (not dj):
                            continue

                        # Again, if the adjacent point has a value, it is in invalid traversal.
                        if self.c_free[i + di, j + dj]:
                            continue

                        tgt_loc = (
                            self.arena.x_mesh[i + di, j + dj],
                            self.arena.y_mesh[i + di, j + dj],
                        )
                        tgt = self.get_node_by_coord(*tgt_loc)

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
        return self.static_obs + self.dynamic_obs

    def reachable_points(self, obstacle_list, epsilon=1, max_val=100):
        """Get the reachable array based on the obstacle_list provided.

        A zero indicates the cell is reachable, and max_val indicates the cell is not reachable.

        """
        p_reach = np.zeros(self.arena.x_mesh.ravel().shape)
        Q = np.block([[self.arena.x_mesh.ravel()], [self.arena.y_mesh.ravel()]])

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
        p_reach = p_reach.reshape(self.arena.x_mesh.shape)

        # Set the boundary elements
        p_reach[0, :] = max_val
        p_reach[-1, :] = max_val
        p_reach[:, 0] = max_val
        p_reach[:, -1] = max_val

        return p_reach

    def reachable_coordinates(self):

        q_reach = np.array(
            [
                (self.arena.x_mesh[i, j], self.arena.y_mesh[i, j])
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

    def get_coordinates(self, node_list):
        node_coords = []
        for v in node_list:
            vert = self.graph.vertex(v)
            node_coords.append(self.graph.vp["pos"][vert])

        return np.array(node_coords)

    def check_line_of_sight(self, x0, y0, xd, yd, v_speed=1, epsilon=1):
        """Check the line of sight between a point q0 and qd, i.e.,
        the agent can move from q0 to qd without the presence of an
        object.

        Args:
            q0 (np.ndarray): current location of agent
            qd (np.ndarray): desired location of agent
            v_speed (float): constant velocity of vehicle
            epsilon (float): addition radius buffer for obstacle

        Returns:
            free_path (bool): if a path exists between q0 and qd.
        """
        if not self.all_obstacles():
            return True

        q0 = np.array([x0, y0]).reshape(-1, 1)
        qd = np.array([xd, yd]).reshape(-1, 1)
        obs_list = np.block([o.as_vec() for o in self.all_obstacles()]).T

        dq = qd - q0
        d_crow = np.linalg.norm(dq)

        v_norm = dq / d_crow

        # Get directional velocities
        v_agent = v_speed * v_norm

        # Keep the to account for moving obstacles
        v_obs = 0 * np.array([[np.cos(0)], [np.sin(0)]])

        # Quadratic equation terms
        a_x = obs_list[:, 0] - q0[0, 0]
        a_y = obs_list[:, 1] - q0[1, 0]

        b_x = v_obs[0, :] - v_agent[0, :]
        b_y = v_obs[1, :] - v_agent[1, :]

        a0 = b_x ** 2 + b_y ** 2
        b0 = 2.0 * a_x * b_x + 2.0 * a_y * b_y
        c0 = a_x ** 2 + a_y ** 2 - (obs_list[:, 2] + epsilon) ** 2

        discriminant = b0 ** 2 - 4.0 * a0 * c0

        # The trajectories intersect if the discriminant is positive
        discriminant = np.where(discriminant > 0.0, discriminant, 0.0)

        t_star_1 = (-b0 - np.sqrt(discriminant)) / (2.0 * a0)
        t_star_2 = (-b0 + np.sqrt(discriminant)) / (2.0 * a0)

        # There are three distinct scenarios:
        # 1. Both positive -- collision is in front of us
        collision_flag = np.logical_and(
            np.logical_and(t_star_1 > 0.0, t_star_2 > 0.0), t_star_1 != t_star_2
        )

        # 2. One positive, one negative -- we are within the bubble
        in_sigma_flag = np.sign(t_star_1) != np.sign(t_star_2)

        # 3. Both negative -- collision is behind us
        no_collision = np.logical_or(
            np.logical_and(t_star_1 < 0.0, t_star_2 < 0.0),
            np.isclose(t_star_1, t_star_2),
        )

        free_path = True
        # Check if the collision if before or after  we reach out desintation
        for idx, collision in enumerate(collision_flag):
            if collision:
                t_star = np.maximum(0, np.minimum(t_star_1[idx], t_star_2[idx]))
                d_star = v_speed * t_star

                if d_crow > d_star:
                    free_path = False
                    return free_path

        return free_path

    def plot_environment(self, ax, nodes_visited, debug=False):
        ax.clear()
        for o in self.all_obstacles():
            o = o.as_vec()
            circle = plt.Circle(o[:2], o[2])
            ax.plot(o[0], o[1], "w+")
            ax.add_artist(circle)
        ax.contour(self.arena.x_mesh, self.arena.y_mesh, self.c_reach)

        if nodes_visited:
            path = self.get_coordinates(nodes_visited)
            ax.plot(path[:, 0], path[:, 1], "r--", alpha=0.5)
            ax.plot(path[0, 0], path[0, 1], "ro", alpha=0.5)
            ax.plot(path[-1, 0], path[-1, 1], "rx", alpha=0.5)

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
