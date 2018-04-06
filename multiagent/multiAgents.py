# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
from game import Agent
from game import Actions
import random
import util
import time
import sys
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        safeDistance = 3
        manhatD = util.manhattanDistance
        # Scare
        s = successorGameState.getScore()
        # Food count
        fC = newFood.count()
        # Closest Food
        if fC == 0:
            F = 0
        elif fC == 1:
            F = manhatD(newFood.asList()[0], newPos)
        else:
            F = min([manhatD(newPos, food) for food in newFood.asList()])
        # closest bad ghost and scared ghost
        badGhostDs = []
        scaredGhostsDs = []
        for ghost in newGhostStates:
            distance = manhatD(newPos, ghost.getPosition())
            if ghost.scaredTimer > 0 :
                scaredGhostsDs.append(distance)
                # print distance*ghost.scaredTimer
            elif distance < safeDistance:
                badGhostDs.append(distance)
        #closest bad ghost
        if len(badGhostDs) == 0:
            bG = 0
        else:
            bG = min(badGhostDs)
            if bG <= 1:
                return -sys.maxint
        if len(scaredGhostsDs) == 0:
            sG = 0
        else:
            sG = min(scaredGhostsDs)

        k = [0, -1000, -10,  5,  -20]
        v = [s, fC,  F, bG, sG]
        pt = sum(k[i]*v[i] for i in range(len(k)))
        return pt

def scoreEvaluationFunction(currentGameState, memory):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.memory = {}


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def miniMax(depth, gameState, agent, prevMove):
            numAgents = gameState.getNumAgents()
            if gameState.isWin():
                return self.evaluationFunction(gameState, self.memory), prevMove
            elif gameState.isLose():
                return self.evaluationFunction(gameState, self.memory), prevMove
            if depth <= 0:
                return self.evaluationFunction(gameState, self.memory), prevMove
            if agent == 0:
                bestMove = None
                maxValue = -sys.maxint
                legalActions = gameState.getLegalActions(agent)
                for action in legalActions:
                    successorState = gameState.generateSuccessor(agent, action)
                    v, move = miniMax(depth, successorState, agent + 1, action)
                    if v > maxValue:
                        maxValue = v
                        bestMove = action
                return maxValue, bestMove
            else:
                minValue = sys.maxint
                bestMove = None
                legalActions = gameState.getLegalActions(agent)
                for action in legalActions:
                    successorState = gameState.generateSuccessor(agent, action)
                    incrDepth = depth
                    if (agent + 1) % numAgents == 0:
                        incrDepth = depth - 1
                    v, move = miniMax(incrDepth, successorState, (agent + 1) % numAgents, action)
                    if v < minValue:
                        minValue = v
                        bestMove = action
                return minValue, bestMove
        maxV, move = miniMax(self.depth, gameState, 0, None)
        return move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(depth, gameState, agent, prevMove, a, b):
            numAgents = gameState.getNumAgents()
            if gameState.isWin():
                return self.evaluationFunction(gameState, self.memory), prevMove
            elif gameState.isLose():
                return self.evaluationFunction(gameState, self.memory), prevMove
            if depth <= 0:
                return self.evaluationFunction(gameState, self.memory), prevMove
            if agent == 0:
                bestMove = None
                maxValue = -sys.maxint
                legalActions = gameState.getLegalActions(agent)
                for action in legalActions:
                    successorState = gameState.generateSuccessor(agent, action)
                    v, move = alphaBeta(depth, successorState, agent + 1, action, a, b)
                    if v > maxValue:
                        maxValue = v
                        bestMove = action
                    if v > b:
                        return v, None
                    a = max(v, a)
                return maxValue, bestMove
            else:
                minValue = sys.maxint
                bestMove = None
                legalActions = gameState.getLegalActions(agent)
                for action in legalActions:
                    successorState = gameState.generateSuccessor(agent, action)
                    incrDepth = depth
                    if (agent + 1) % numAgents == 0:
                        incrDepth = depth - 1
                    v, move = alphaBeta(incrDepth, successorState, (agent + 1) % numAgents, action, a, b)
                    if v < minValue:
                        minValue = v
                        bestMove = action
                    if v < a:
                        return v, None
                    b = min(v, b)
                return minValue, bestMove

        v, move = alphaBeta(self.depth,gameState,0, None, -sys.maxint, sys.maxint)
        return move



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(depth, gameState, agent, prevMove):
            numAgents = gameState.getNumAgents()
            if gameState.isWin():
                return self.evaluationFunction(gameState, self.memory), prevMove
            elif gameState.isLose():
                return self.evaluationFunction(gameState, self.memory), prevMove
            if depth <= 0:
                return self.evaluationFunction(gameState, self.memory), prevMove
            if agent == 0:
                bestMove = None
                maxValue = -sys.float_info.max
                legalActions = gameState.getLegalActions(agent)
                for action in legalActions:
                    successorState = gameState.generateSuccessor(agent, action)
                    v, move = expectiMax(depth, successorState, agent + 1, action)
                    if v > maxValue:
                        maxValue = v
                        bestMove = action
                return maxValue, bestMove
            else:
                expectiValue = 0.0
                legalActions = gameState.getLegalActions(agent)
                for action in legalActions:
                    successorState = gameState.generateSuccessor(agent, action)
                    incrDepth = depth
                    if (agent + 1) % numAgents == 0:
                        incrDepth = depth - 1
                    v, move = expectiMax(incrDepth, successorState, (agent + 1) % numAgents, action)
                    expectiValue += 1.0/len(legalActions) * v
                return expectiValue, None

        try:
            self.memory["numCaplse"] = len(gameState.getCapsules())
            newGhostStates = gameState.getGhostStates()
            self.memory["hasScaredGhost"] = sum([ghostState.scaredTimer for ghostState in newGhostStates])
        except AttributeError:
            self.memory["numCaplse"] = 0
            self.memory["hasScaredGhost"] = 0
        maxV, move = expectiMax(self.depth, gameState, 0, None)
        return move

def betterEvaluationFunction(currentGameState, memory):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      Modifications:
        Imported search, searchAgent from proj1
        Added self.memory = \{\} to multiAgentSearchAgent to decrease runtime

      Variables:
        s  - curerentGameState.getScore()
        fC - foodCount - number of remaining food pellet
        F  - shortest food maze distance in 3 cases
                1) (fC=0) F = 0
                2) (fC=1) F = mazeDistance to last food
                3) (fc>1) F = mazeDistance to closest food + closestfood to
                                2nd closest
        bG - mazeDistance to closest ghost. bG=0 if there are none.
        sG - mazeDistance to closest scared ghost. sG=0 if there are none.
        C  - mazeDistance to closest capsule.
                C > 0 if there's capulse & ghost within 10 manhatD
                C = -10000.0 if current state has 1 less than original state

        vector K = [500.0, -10000.0, -100.0,  5, -20.0, -20.0]
      Strategy:
        k = [500.0, -10000.0, -100.0,  5, -20.0, -20.0]
        v = [s, fC,  F, bG, sG, C]
        final evaluation = k (dot) v

        1. Prioritize decreasing fC unless pacman can get a Capsule
        2. Find Path for 2 consecutively closest food
        3. Run away from ghost if ghost can eat pacman
        4. Always go for scared ghost

    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()
    foodPositions = newFood.asList()
    "*** YOUR CODE HERE ***"
    manhatD = util.manhattanDistance
    mazeD = mazeDistance
    # Score
    s = currentGameState.getScore()
    # Food left
    fC = newFood.count()
    # Closest Food
    if currentGameState.isWin():
        fC = 0
        F = 0
    elif fC == 1:
        prob = AnyFoodSearchProblem(currentGameState)
        F = len(ucs(prob))
    else:
        # bound list of food for efficiency, also decreased avg score, not worth
        distanceBound = 5
        """ Return list of food within bound"""
        def foodInBound(pos,foodGrid, bound):
            lst = []
            for x in range(foodGrid.width):
                for y in range(foodGrid.height):
                    if abs(x-pos[0]) + abs(y-pos[1]) < bound:
                        if foodGrid[x][y]:
                            lst.append((x,y))
            return lst
        boundFoodPositions = []
        while len(boundFoodPositions) < 2:
            boundFoodPositions = foodInBound(newPos,newFood,distanceBound)
            distanceBound *=2

        minFood = None
        minD = sys.maxint
        for foodPos in boundFoodPositions:
            if (newPos, foodPos) not in memory.keys():
                foodD = mazeD(newPos, foodPos, currentGameState)
                memory[(newPos, foodPos)] = foodD
                memory[(foodPos, newPos)] = foodD
            foodD = memory[(newPos, foodPos)]
            if foodD < minD:
                minD = foodD
                minFood = foodPos
        foodPositions.remove(minFood)
        minFood2 = None
        minD2 = sys.maxint
        for foodPos2 in boundFoodPositions:
            if (minFood, foodPos2) not in memory.keys():
                foodD2 =mazeD(minFood, foodPos2, currentGameState)
                memory[(minFood, foodPos2)] = foodD2
                memory[(foodPos2, minFood)] = foodD2
            foodD2 = memory[(minFood, foodPos2)]
            if foodD2 < minD2:
                minD2 = foodD2
                minFood2 = foodPos2
        F = minD + minD2
    # closest bad ghost and scared ghost
    badGhostDs = []
    scaredGhostsDs = []
    for ghost in newGhostStates:
        ghostPos = (int(ghost.getPosition()[0]),int(ghost.getPosition()[1]))
        distance = manhatD(newPos, ghostPos)
        if ghost.scaredTimer > 0 :
            scaredGhostsDs.append(distance)
        else:
            badGhostDs.append(distance)
    if len(badGhostDs) == 0:
        bG = 0
    else:
        bG = min(badGhostDs)
        if bG <= 1:
            return -sys.maxint
    if len(scaredGhostsDs) == 0:
        sG = 0
    else:
        sG = min(scaredGhostsDs)
    # Eating calsule
    C = 0
    capDs = []
    if len(newCapsules) > 0 :
        for capPos in newCapsules:
            capD = mazeD(newPos,capPos,currentGameState)
            memory[(newPos, capPos)] = capD
            memory[(capPos, newPos)] = capD
            capDs.append(capD)
        C = min(capDs)
    try:
        if len(currentGameState.getCapsules()) < memory["numCaplse"] and memory["hasScaredGhost"] == 0:
            C = -100000.0
    except AttributeError:
        pass
    k = [500.0, -10000.0, -100.0,  20, -10000.0, -20.0]
    v = [  s,       fC,      F, bG,    sG,     C]
    pt = sum(k[i]*v[i] for i in range(len(k)))
    return pt

# Abbreviation
better = betterEvaluationFunction

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        self.memory = {}
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        foodPos = gameState.getFood().asList()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return ucs(problem)


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return state in self.food.asList()

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(astar(prob, manhattanHeuristic))



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    closed = set()
    fringe = util.Stack()
    fringe.push([problem.getStartState(),[]])

    while True:
        if fringe.isEmpty():
            print "Cannot reach goal state."
            return None
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if node[0] not in closed:
            closed.add(node[0])
            for successor, action, cost in problem.getSuccessors(node[0]):
                accumActions = node[1][:]
                accumActions.append(action)
                childNode = [successor, accumActions]
                fringe.push(childNode)



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed = set()
    fringe = util.Queue()
    fringe.push([problem.getStartState(),[]])

    while True:
        if fringe.isEmpty():
            print "Cannot reach goal state."
            return None
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if node[0] not in closed:
            closed.add(node[0])
            for successor, action, cost in problem.getSuccessors(node[0]):
                accumActions = node[1][:]
                accumActions.append(action)
                childNode = [successor, accumActions]
                fringe.push(childNode)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    closed = set()
    fringe = util.PriorityQueue()
    fringe.push([problem.getStartState(),[]],0)

    while True:
        if fringe.isEmpty():
            print "Cannot reach goal state."
            return None
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if node[0] not in closed:
            closed.add(node[0])
            for successor, action, cost in problem.getSuccessors(node[0]):
                accumActions = node[1][:]
                accumActions.append(action)
                childNode = [successor, accumActions]
                g = problem.getCostOfActions(accumActions)
                fringe.push(childNode, g)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closed = set()
    fringe = util.PriorityQueue()
    fringe.push([problem.getStartState(), []], 0)

    while True:
        if fringe.isEmpty():
            print "Cannot reach goal state."
            return None
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if node[0] not in closed:
            closed.add(node[0])
            for successor, action, cost in problem.getSuccessors(node[0]):
                accumActions = node[1][:]
                accumActions.append(action)
                childNode = [successor, accumActions]
                g = problem.getCostOfActions(accumActions)
                h = heuristic(successor, problem)
                f = g + h
                fringe.push(childNode, f)

def greedySearch(problem, heuristic=nullHeuristic):
    closed = set()
    fringe = util.PriorityQueue()
    fringe.push([problem.getStartState(),[]],0)
    while True:
        if fringe.isEmpty():
            print "Cannot reach goal state."
            return None
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1]
        if node[0] not in closed:
            closed.add(node[0])
            for successor, action, cost in problem.getSuccessors(node[0]):
                accumActions = node[1][:]
                accumActions.append(action)
                childNode = [successor, accumActions]
                h = heuristic(successor, problem)
                fringe.push(h)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
greedy = greedySearch
