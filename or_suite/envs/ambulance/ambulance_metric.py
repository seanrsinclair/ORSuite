'''
Implementation of a basic RL environment for continuous spaces.
Includes three test problems which were used in generating the figures.
'''

import numpy as np
import gym
from gym import spaces
import math
from .. import env_configs

#------------------------------------------------------------------------------
'''An ambulance environment over [0,1].  An agent interacts through the environment
by picking a location to station the ambulance.  Then a patient arrives and the ambulance
most go and serve the arrival, paying a cost of travel.'''



class AmbulanceEnvironment(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the arrivals are always uniformly distributed
  """

  metadata = {'render.modes': ['human']}


  def __init__(self, config = env_configs.ambulance_metric_default_config):
        '''
        For a more detailed description of each parameter, see the readme file
        
        epLen - number of time steps
        arrival_dist - arrival distribution for calls over the space [0,1]
        alpha - parameter for proportional difference in costs
        starting_state - a list containing the starting locations for each ambulance
        num_ambulance - the number of ambulances in the environment 
        '''
        super(AmbulanceEnvironment, self).__init__()

        self.config = config
        self.epLen = config['epLen']
        self.alpha = config['alpha']
        self.starting_state = config['starting_state']
        self.state = np.array(self.starting_state, dtype=np.float32)
        self.timestep = 0
        self.num_ambulance = config['num_ambulance']
        self.arrival_dist = config['arrival_dist']

        self.viewer = None

        # The action space is a box with each ambulances location between 0 and 1
        self.action_space = spaces.Box(low=0, high=1,
                                        shape=(self.num_ambulance,), dtype=np.float32)

        # The observation space is a box with each ambulances location between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(self.num_ambulance,), dtype=np.float32)

  def reset(self):
        """
        Reinitializes variables and returns the starting state
        """
        # Initialize the timestep
        self.timestep = 0
        self.state = self.starting_state

        return self.starting_state

  def get_config(self):
        return self.config

  def step(self, action):
        '''
        Move one step in the environment

        Args:
        action - float list - list of locations in [0,1] the same length as the 
        number of ambulances, where each entry i in the list corresponds to the 
        chosen location for ambulance i
        Returns:
            reward - float - reward based on the action chosen
            newState - float list - new state
            done - 0/1 - flag for end of the episode
        '''
        old_state = np.array(self.state)

        # The location of the new arrival is chosen randomly from the arrivals 
        # distribution arrival_dist
        new_arrival = self.arrival_dist(self.timestep)

        # The closest ambulance to the call is found using the l-1 distance
        close_index = np.argmin(np.abs(old_state - new_arrival))

        # Update the state of the system according to the action taken and change 
        # the location of the closest ambulance to the call to the call location
        action = np.array(action, dtype=np.float32)
        new_state = action.copy()
        new_state[close_index] = new_arrival

        # print("Old", old_state)
        # print("Action", action)
        # print("Close Index", close_index)
        # print("New Arrival", new_arrival)
        # print("New", new_state)

        # The reward is a linear combination of the distance traveled to the action
        # and the distance traveled to the call
        # alpha controls the tradeoff between cost to travel between arrivals and 
        # cost to travel to a call
        # The reward is negated so that maximizing it will minimize the distance

        print("alpha", self.alpha)

        reward = -1 * (self.alpha * np.sum(np.abs(old_state - action)) + (1 - self.alpha) * np.sum(np.abs(action - new_state)))
        
        print("First", np.sum(np.abs(old_state - action)))
        print("action", action)
        print("new state", new_state)
        print("Second", np.max(np.abs(action - new_state)))
        print("Reward", reward)
        
        # The info dictionary is used to pass the location of the most recent arrival
        # so it can be used by the agent
        info = {'arrival' : new_arrival}

        if self.timestep <= self.epLen:
            pContinue = True
        else:
            pContinue = False

        self.state = new_state
        self.timestep += 1

        return self.state, reward,  pContinue, info


  # def render(self, mode='console'):
  #   if mode != 'console':
  #     raise NotImplementedError()

  def render(self, mode='human'):
    screen_width = 600
    screen_height = 400

    #world_width = self.x_threshold * 2
    #scale = screen_width/world_width
    #carty = 100  # TOP OF CART
    #polewidth = 10.0
    #polelen = scale * (2 * self.length)
    #cartwidth = 50.0
    #cartheight = 30.0

    if self.viewer is None:
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(screen_width, screen_height)
        # l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        # axleoffset = cartheight / 4.0
        # cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        # self.carttrans = rendering.Transform()
        # cart.add_attr(self.carttrans)
        line_0 = (50, 100)
        line_1 = (550, 100)
        self.number_line = rendering.Line(line_0, line_1)
        self.number_line.set_color(0, 0, 0)
        self.viewer.add_geom(self.number_line)

        # self.viewer.add_geom(cart)
        # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        # pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        # pole.set_color(.8, .6, .4)
        # self.poletrans = rendering.Transform(translation=(0, axleoffset))
        # pole.add_attr(self.poletrans)
        # pole.add_attr(self.carttrans)
        # self.viewer.add_geom(pole)
        # self.axle = rendering.make_circle(polewidth/2)
        # self.axle.add_attr(self.poletrans)
        # self.axle.add_attr(self.carttrans)
        # self.axle.set_color(.5, .5, .8)
        # self.viewer.add_geom(self.axle)
        # self.track = rendering.Line((0, carty), (screen_width, carty))
        # self.track.set_color(0, 0, 0)
        # self.viewer.add_geom(self.track)

        # self._pole_geom = pole

    if self.state is None:
        return None

    # Edit the pole polygon vertex
    # pole = self._pole_geom
    # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
    # pole.v = [(l, b), (l, t), (r, t), (r, b)]

    # x = self.state
    # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    # self.carttrans.set_translation(cartx, carty)
    # self.poletrans.set_rotation(-x[2])

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    pass

