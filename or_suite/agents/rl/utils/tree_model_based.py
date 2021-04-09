import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
''' Implementation of a tree structured used in the Adaptive Discretization Algorithm'''








import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

from or_suite.agents.rl.utils.bounds_utils import bounds_contains, split_bounds
from or_suite.agents.rl.utils.tree import Node, Tree




class MBNode(Node):

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



    def __init__(self, bounds, depth, qVal, rEst, pEst, num_visits):

        self.dim = len(bounds)
        # print(bounds)
        self.radius = (bounds[:, 1] - bounds[:, 0]).max() / 2.0
        # print(self.radius)
        assert self.radius > 0.0

        self.bounds = bounds
        self.depth = depth
        self.qVal = qVal
        self.rEst = rEst
        self.pEst = pEst
        self.num_visits = num_visits

        self.children = []


        # Splits a node
    def split_node(self, inherit_flag = True, value = 1):
        child_bounds = split_bounds(self.bounds)
        for bounds in child_bounds:
            if inherit_flag:
                self.children.append(
                    MBNode(bounds, self.depth+1, self.qVal, self.rEst, self.pEst, self.num_visits)
                )
            else:
                # TODO: Update transitions here?
                self.children.append(
                    MBNode(bounds, self.depth+1, value, 0, 0*np.asarray(self.pEst), 0)
                )

        return self.children





class MBTree(Tree):

    """
        Tree-based partition of an l-infinity ball in R^d.
        Each node is of type TreeNode.
    """

    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen, state_dim, action_dim):
        self.dim = state_dim+action_dim
        self.epLen = epLen
        self.state_dim = state_dim

        bounds = np.asarray([[0.0,1.0] for _ in range(self.dim)])

        self.head = MBNode(bounds, 0, epLen, 0, [0.0], 0)
        
        self.state_leaves = [[0.5 for _ in range(self.state_dim)]]
        self.leaves = [self.head]
        self.vEst = [self.epLen]


    def get_leaves(self):
        return self.leaves

    def split_node(self, node, inherit_flag = True, value = 1):

        self.leaves.remove(node)
        children = node.split_bounds(inherit_flag, value)
        self.leaves = self.leaves + children

        # Determines if we also need to adjust the state_leaves and carry those
        # estimates down as well

        # Gets one of their state value
        child_1_bounds = children[0].bounds
        child_1_radius = (child_1_bounds[:, 1] - child_1_bounds[:, 0]).max() / 2.0
        child_1_state = child_1_bounds[:self.state_dim, 0] + child_1_radius

        # if np.min(np.max(np.abs(state_1 - child_state_1), np.abs(state_2 - child_state_2))) >= child_1_radius
        if np.min(np.abs(np.asarray(self.state_leaves) - child_1_state)) >= child_1_radius:
            # print('Adjusting the induced state partition')
            # print('Current node state: ' + str(node.state_val))
            # print('Child state: ' + str(child_1_state))
            # print('Current leaves: ' + str(self.state_leaves))

            node_radius = (node.bounds[:, 1] - node.bounds[:, 0]).max() / 2.0
            node_state = node.bounds[:self.state_dim, 0] + node_radius

            # print('Getting parents index!')
            # print(self.state_leaves)
            parent_index = self.state_leaves.index(node_state)
            parent_vEst = self.vEst[parent_index]

            self.state_leaves.pop(parent_index)
            self.vEst.pop(parent_index)

            # will be appending duplicate numbers here

            # self.state_leaves.append(child.state_val)

            # append(children[0].state_val(0), children[0].state_val(1))
            num_add = 0
            for child in node.children:
                child_radius = (child.bounds[:,1] - child.bounds[:,0]).max() / 2.0
                child_state = child.bounds[:self.state_dim, 0] + child_radius
                if child_state not in self.state_leaves:
                    num_add += 1
                    self.state_leaves.append(child_state)
                    self.vEst.append(parent_vEst)

            # print('Checking lengths: ')
            # print(len(self.state_leaves))
            # print(len(self.vEst))
            # Lastly we need to adjust the transition kernel estimates from the previous tree
            if timestep >= 1:
                previous_tree.update_transitions_after_split(parent_index, num_add)


            # Need to remove parent's state value from state_leaves,
            # add in the state values for the children
            # copy over the estimate of the value function
            # also copy over the estimate of the transition function
        # print(self.state_leaves)
        return children


    def update_transitions_after_split(self, parent_index, num_children):
        # print('Adjusting transitions at previous timestep')
        # print('Number of leaves: ' + str(len(self.tree_leaves)))
        # print('Printing out length for each leaf!')
        # for node in self.tree_leaves:
        #     print(len(node.pEst))
        #     print(node.pEst)
        #     print(node)

        # print('Starting to adjust')
        for node in self.leaves:
            # Adjust node.pEst
            # Should not just be copy pasting here....
            # print('Start adjust for a node!')
            # print(node)
            # print(len(node.pEst))
            # print(node.pEst)
            pEst_parent = node.pEst[parent_index]
            node.pEst.pop(parent_index)
            # print(len(node.pEst))
            # print('Adding on entries now!')
            for _ in range(num_children):
                node.pEst.append(pEst_parent / num_children)
            # print(len(node.pEst))
            # print(node.pEst)
            # print('Done')