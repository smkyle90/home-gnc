# import graph_tool.all as gt
# import numpy as np

# """
# https://www.researchgate.net/publication/221077776_Anytime_Motion_Planning_using_the_RRT
# """


# def rrt_star(q_init, N):
# 	g = gt.Graph()


# 	for i in range(N):

# 		q_sample = sample_node()

# 		q_near = nearest_node(g, q_sample)

# 		x_reach, u_new = steer(q_near, q_sample)
# 		q_reach = x_reach[-1]

# 		if obstacle_free(x_reach):

# 			Q_near = all_nearest(g, q_reach, num_verts)

# 			q_min = choose_parent_node(Q_near, q_near, x_reach)

# 			g = add_node(q_min, q_reach, g)

# 			g = rewire(g, Q_near, q_min, q_reach)


# def obstacle_free(path):
# 	return True

# def choose_parent_node(Q_near, q_near, x_reach):
# 	q_reach = x_reach[-1]
# 	q_min = q_near
# 	c_min = cost(q_near) + cost(q_reach)

# 	for q_near in Q_near:


# def rewire():
