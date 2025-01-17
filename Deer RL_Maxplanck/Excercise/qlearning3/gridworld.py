import random
import sys
import mdp
import environment
import util
import optparse

class Gridworld(mdp.MarkovDecisionProcess):

  def __init__(self, grid):
    # layout
    self.grid = grid
    self.rows = len(grid)
    self.cols = len(grid[0])
    self.terminalState = (-1, -1)

    # parameters
    self.livingReward = 0.0
    self.noise = 0.2

  def setLivingReward(self, reward):
    """
    The (negative) reward for exiting "normal" states.

    Note that in the R+N text, this reward is on entering
    a state and therefore is not clearly part of the state's
    future rewards.
    """
    self.livingReward = reward

  def setNoise(self, noise):
    """
    The probability of moving in an unintended direction.
    """
    self.noise = noise


  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.

    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    if state == self.terminalState:
      return ()
    row, col = state
    if type(self.grid[row][col]) == int:
      return ('exit',)
    return ('north','west','south','east')


  def getStates(self):
    """
    Return list of all states.
    """
    states = [self.terminalState]  # The true terminal state.
    for row in range(self.rows):
      for col in range(self.cols):
        if self.grid[row][col] != '#':
          states.append((row, col))
    return states

  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.

    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    if state == self.terminalState:
      return 0.0
    row, col = state
    cell = self.grid[row][col]
    if type(cell) == int or type(cell) == float:
      return cell
    return self.livingReward

  def getStartState(self):
    for row in range(self.rows):
      for col in range(self.cols):
        if self.grid[row][col] == 'S':
          return (row, col)
    raise 'Grid has no start state'

  def isTerminal(self, state):
    """
    Only the (-1, -1) state is *actually* a terminal state.
    The other "exit" states are technically non-terminals with
    a single action "exit" which leads to the true terminal state.
    This convention is to make the grids line up with the examples
    in the R+N textbook.
    """
    return state == self.terminalState


  def getTransitionStatesAndProbs(self, state, action):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.
    """

    if action not in self.getPossibleActions(state):
      raise "Illegal action!"

    if state == self.terminalState:
      return []

    row, col = state

    if type(self.grid[row][col]) == int or type(self.grid[row][col]) == float:
      return [(self.terminalState, 1.0)]

    successors = []

    northState = (self.__isAllowed(row-1,col) and (row-1,col)) or state
    westState = (self.__isAllowed(row,col-1) and (row,col-1)) or state
    southState = (self.__isAllowed(row+1,col) and (row+1,col)) or state
    eastState = (self.__isAllowed(row,col+1) and (row,col+1)) or state

    if action == 'north' or action == 'south':
      if action == 'north':
        successors.append((northState,1-self.noise))
      else:
        successors.append((southState,1-self.noise))

      massLeft = self.noise
      successors.append((westState,massLeft/2.0))
      successors.append((eastState,massLeft/2.0))

    if action == 'west' or action == 'east':
      if action == 'west':
        successors.append((westState,1-self.noise))
      else:
        successors.append((eastState,1-self.noise))

      massLeft = self.noise
      successors.append((northState,massLeft/2.0))
      successors.append((southState,massLeft/2.0))

    successors = self.__aggregate(successors)

    return successors

  def __aggregate(self, statesAndProbs):
    counter = util.Counter()
    for state, prob in statesAndProbs:
      counter.incrementCount(state, prob)
    newStatesAndProbs = []
    # print counter
    for state, prob in list(counter.items()):
      newStatesAndProbs.append((state, prob))
    return newStatesAndProbs

  def __isAllowed(self, row, col):
    if row < 0 or row >= self.rows: return False
    if col < 0 or col >= self.cols: return False
    return self.grid[row][col] != '#'



class GridworldEnvironment(environment.Environment):

  def __init__(self, gridWorld):
    self.gridWorld = gridWorld
    self.reset()

  def getCurrentState(self):
    return self.state

  def getPossibleActions(self, state):
    return self.gridWorld.getPossibleActions(state)

  def doAction(self, action):
    successors = self.gridWorld.getTransitionStatesAndProbs(self.state, action)
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.'
      if rand < sum:
        reward = self.gridWorld.getReward(state, action, nextState)
        self.state = nextState
        return (nextState, reward)
    raise 'Total transition probability less than one; sample failure.'

  def reset(self):
    self.state = self.gridWorld.getStartState()


def getCliffGrid():
  grid = [[' ',' ',' ',' ',' '],
          ['S',' ',' ',' ',10],
          [-100,-100, -100, -100, -100]]
  return Gridworld(grid)

def getCliffGrid2():
  grid = [[' ',' ',' ',' ',' '],
          [8,'S',' ',' ',10],
          [-100,-100, -100, -100, -100]]
  return Gridworld(grid)

def getDiscountGrid():
  grid = [[' ',' ',' ',' ',' '],
          [' ','#',' ',' ',' '],
          [' ','#', 1,'#', 10],
          ['S',' ',' ',' ',' '],
          [-10,-10, -10, -10, -10]]
  return Gridworld(grid)

def getBridgeGrid():
  grid = [[ '#',-100, -100, -100, -100, -100, '#'],
          [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
          [ '#',-100, -100, -100, -100, -100, '#']]
  return Gridworld(grid)

def getBookGrid():
  grid = [[' ',' ',' ',+1],
          [' ','#',' ',-1],
          ['S',' ',' ',' ']]
  return Gridworld(grid)

def getMazeGrid():
  grid = [[' ',' ',' ',+1],
          ['#','#',' ','#'],
          [' ','#',' ',' '],
          [' ','#','#',' '],
          ['S',' ',' ',' ']]
  return Gridworld(grid)



def getUserAction(state, actionFunction):
  """
  Get an action from the user (rather than the agent).

  Used for debugging and lecture demos.
  """
  import graphicsUtils
  action = None
  while True:
    keys = graphicsUtils.wait_for_keys()
    if 'Up' in keys: action = 'north'
    if 'Down' in keys: action = 'south'
    if 'Left' in keys: action = 'west'
    if 'Right' in keys: action = 'east'
    if 'q' in keys: sys.exit(0)
    if action == None: continue
    break
  actions = actionFunction(state)
  if action not in actions:
    action = actions[0]
  return action

def printString(x):
  print(x)

def runEpisode(agent, environment, discount, decision, display, message, pause, episode):
  returns = 0
  totalDiscount = 1.0
  environment.reset()
  agent.reset() # new
  message("BEGINNING EPISODE: "+str(episode)+"\n")
  while True:

    # DISPLAY CURRENT STATE
    state = environment.getCurrentState()
    display(state)
    pause()

    # END IF IN A TERMINAL STATE
    actions = environment.getPossibleActions(state)
    if len(actions) == 0:
      message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
      return returns

    # GET ACTION (USUALLY FROM AGENT)
    action = decision(state)
    if action == None:
      raise 'Error: Agent returned None action'

    # EXECUTE ACTION
    nextState, reward = environment.doAction(action)
    message("Started in state: "+str(state)+
            "\nTook action: "+str(action)+
            "\nEnded in state: "+str(nextState)+
            "\nGot reward: "+str(reward)+"\n")

    # UPDATE LEARNER
    agent.update(state, action, nextState, reward)
    returns += reward * totalDiscount
    totalDiscount *= discount



def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="I", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=0,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="BookGrid",
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent (for lecture)')

    opts, args = optParser.parse_args()

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
      opts.pause = False
      opts.manual = False

    if opts.manual:
      opts.pause = False

    return opts


if __name__ == '__main__':

  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  import gridworld
  mdpFunction = getattr(gridworld, "get"+opts.grid)
  mdp = mdpFunction()
  mdp.setLivingReward(opts.livingReward)
  mdp.setNoise(opts.noise)
  env = gridworld.GridworldEnvironment(mdp)


  ###########################
  # GET THE DISPLAY ADAPTER
  ###########################

  import textGridworldDisplay
  display = textGridworldDisplay.TextGridworldDisplay(mdp)
  if not opts.textDisplay:
    import graphicsGridworldDisplay
    display = graphicsGridworldDisplay.GraphicsGridworldDisplay(mdp, opts.gridSize, opts.speed)
  display.start()

  ###########################
  # GET THE AGENT
  ###########################

  import agent
  a = None
  if opts.agent == 'value':
    a = agent.ValueIterationAgent(mdp, opts.discount, opts.iters)
  elif opts.agent == 'q':
    a = agent.QLearningAgent(env.getPossibleActions, opts.discount, opts.learningRate, opts.epsilon)
  elif opts.agent == 'random':
    # No reason to use the random agent without episodes
    if opts.episodes == 0:
      opts.episodes = 1
    a = agent.RandomAgent(mdp.getPossibleActions)
  else:
    raise 'Unknown agent type: '+opts.agent

  ###########################
  # RUN EPISODES
  ###########################

  # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
  if opts.agent == 'value':
    display.displayValues(a, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    display.pause()
    display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    display.pause()

  # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
  displayCallback = lambda x: None
  if not opts.quiet:
    if opts.agent == 'random': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
    if opts.agent == 'value': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
    if opts.agent == 'q': displayCallback = lambda state: display.displayQValues(a, state, "CURRENT Q-VALUES")

  messageCallback = lambda x: printString(x)
  if opts.quiet:
    messageCallback = lambda x: None

  # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
  pauseCallback = lambda : None
  if opts.pause:
    pauseCallback = lambda : display.pause()

  # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)
  decisionCallback = a.getAction
  if opts.manual:
    decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)

  # RUN EPISODES
  if opts.episodes > 0:
    print()
    print("RUNNING", opts.episodes, "EPISODES")
    print()
  returns = 0
  for episode in range(1, opts.episodes+1):
    returns += runEpisode(a, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
  if opts.episodes > 0:
    print()
    print("AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes))
    print()
    print()

  # DISPLAY POST-LEARNING VALUES / Q-VALUES
  if opts.agent == 'q' and not opts.manual:
    display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.episodes)+" EPISODES")
    display.pause()
    display.displayValues(a, message = "VALUES AFTER "+str(opts.episodes)+" EPISODES")
    display.pause()
