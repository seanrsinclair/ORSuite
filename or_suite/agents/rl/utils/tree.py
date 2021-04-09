import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

from or_suite.agents.rl.utils.bounds_utils import bounds_contains, split_bounds





class Node():

    """
    Node representing an l-infinity ball in R^d, that points
    to sub-balls (defined via node children).
    Stores a value for the q_estimate, a number of visits, and 
    
        TODO: (possibly) rewards and transition probability to a list of other nodes.


    This class is used to represent (and store data about)
    a tuple (state, action, stage) = (x, a, h).


    Parameters
    ----------
    bounds : numpy.ndarray
        Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
        representing the cartesian product in R^d:
        [x0, y0] X [x1, y1] X ... X [xd, yd]
    depth: int
        Node depth, root is at depth 0.
    qVal : double, default: 0
        Initial node Q value
    num_visits : int, default = 0
        Number of visits to the node.
    """



    def __init__(self, bounds, depth, qVal, num_visits):

        self.dim = len(bounds)
        # print(bounds)
        self.radius = (bounds[:, 1] - bounds[:, 0]).max() / 2.0
        # print(self.radius)
        assert self.radius > 0.0

        self.bounds = bounds
        self.depth = depth
        self.qVal = qVal
        self.num_visits = num_visits

        self.children = []


    def is_leaf(self):
        return len(self.children) == None

    def contains(self, state):
        return bounds_contains(self.bounds, state)



    # Splits a node
    def split_node(self):
        child_bounds = split_bounds(self.bounds)
        for bounds in child_bounds:
            self.children.append(
                Node(bounds, self.depth+1, self.qVal, self.num_visits)
            )

        return self.children




class Tree():

    """
        Tree-based partition of an l-infinity ball in R^d.
        Each node is of type TreeNode.
    """

    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen, dim):
        self.dim = dim

        bounds = np.asarray([[0.0,1.0] for _ in range(dim)])

        self.head = Node(bounds, 0, epLen, 0)
        self.epLen = epLen
        
    # Returns the head of the tree
    def get_head(self):
        return self.head

    def get_max(self, node = None, root = True):
        if root:
            node = self.head

        if len(node.children) == 0:
            return node.qVal
        else:
            return np.max([self.get_max(child, False) for child in node.children])


    def get_min(self, node = None, root = True):
        if root:
            node = self.head


        if len(node.children) == 0:
            return node.qVal
        else:
            return np.min([self.get_min(child, False) for child in node.children])

    # TODO: Might need to make some edits to this
    def plot(self, figname = 'tree plot', colormap_name = 'cool', max_value = 10, node=None, root=True,):
        if root:
            assert self.dim == 2, "Plot only available for 2-dimensional spaces."
            plt.figure(fignum)
        
        if node.is_leaf():
            x0, x1 = node.bounds[0, :]
            y_1, y_1 = node.bounds[1, :]
            colormap_fn = plt.get_cmap(colormap_name)
            color = colormap_fn(node.qVal / max_value)
            rectangle = plt.Rectangle((x0, y0), x1-x0, y1-y0, ec='black', color=color)
            plt.gca().add_patch(rectangle)
            plt.axis('scaled')
        else:
            for cc in node.children:
                self.plot(max_value = max_value, colormap_name = colormap_name, node=cc, root=False)


    # Recursive method which gets number of subchildren
    def get_num_balls(self, node = None, root = True):
        if root:
            node = self.head

        num_balls = 1
        for child in node.children:
            num_balls += self.get_num_balls(child, False)
        return num_balls



    def get_active_ball(self, state, node = None, root = True):
        if root:
            node = self.head

        if len(node.children) == 0:
            return node, node.qVal
        
        else:
            best_node = node
            best_qVal = node.qVal

            for child in node.children:
                if child.contains(state):
                    nn, nn_qVal = self.get_active_ball(state, child, False)
                    if nn_qVal >= best_qVal:
                        best_node, best_qVal = nn, nn_qVal
            return best_node, best_qVal