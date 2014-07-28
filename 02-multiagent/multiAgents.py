# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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

        oldFood = currentGameState.getFood()
        dist = 1E9
        
        for ghost in newGhostStates:
            if dist> manhattanDistance( ghost.getPosition(), newPos) and  manhattanDistance( ghost.getPosition(), newPos) < ghost.scaredTimer:
                dist =  manhattanDistance( ghost.getPosition(), newPos)
        if dist != 1E9:
            return 1.0/dist   
        
        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) <= 1:
                return 0 
        if oldFood.count() == newFood.count():

            return 1.0/manhattanDistance(newFood.asList()[0],newPos)
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def minimax(self, gamestate, thisdepth, actor):
        numActors = gamestate.getNumAgents()
        if self.depth == thisdepth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
        if actor == 0:
            score = -1E9
            for paction in gamestate.getLegalActions(0):
                newState= gamestate.generateSuccessor(0,paction)
                score = max(score, self.minimax(newState,thisdepth, actor +1))
            return score
        if actor == numActors:
            return self.minimax(gamestate,thisdepth +1, 0)
        score = 1E9
        for gaction in gamestate.getLegalActions(actor):
            newState =gamestate.generateSuccessor(actor,gaction)
            score = min(score,self.minimax(newState, thisdepth,actor +1))
        return score

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
        """
        "*** YOUR CODE HERE ***"
        ghosts = gameState.getNumAgents()
        score = -1E9
        action = Directions.STOP
        for paction in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0,paction)
            tempscore= self.minimax(newState,0, 1)
            if tempscore > score:
                score = tempscore
                action = paction
        return action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def minimax(self, gamestate, thisdepth, actor, a, b):
        numActors = gamestate.getNumAgents()
        if self.depth == thisdepth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
            
        if actor == 0:
            score = -1E9
            for paction in gamestate.getLegalActions(0):
                newState= gamestate.generateSuccessor(0,paction)
                tempscore = self.minimax(newState,thisdepth, actor +1,a,b)
                if tempscore > score:
                    score = tempscore
                if tempscore >a:
                    a = tempscore
                if score > b:
                   return score
            return score
        if actor == numActors:
            return self.minimax(gamestate,thisdepth +1, 0,a,b)
        score = 1E9
        for gaction in gamestate.getLegalActions(actor):
            newState =gamestate.generateSuccessor(actor,gaction)
            tempscore = self.minimax(newState,thisdepth, actor +1,a,b)
            if tempscore < score:
                score = tempscore
            if tempscore < b:
                b = tempscore
            if score < a:
                return score
        return score


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        ghosts = gameState.getNumAgents()
        score = -1E9
        action = Directions.STOP
        for paction in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0,paction)
            tempscore= self.minimax(newState,0, 1,score,1E9)
            if tempscore > score:
                score = tempscore
                action = paction
        return action
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gamestate, thisdepth, actor):
        numActors = gamestate.getNumAgents()
        if self.depth == thisdepth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate)
        if actor == 0:
            score = -1E9
            for paction in gamestate.getLegalActions(0):
                newState= gamestate.generateSuccessor(0,paction)
                score = max(score, self.expectimax(newState,thisdepth, actor +1))
            return score
        if actor == numActors:
            return self.expectimax(gamestate,thisdepth +1, 0)
        score = 0
        for gaction in gamestate.getLegalActions(actor):
            newState =gamestate.generateSuccessor(actor,gaction)
            score += self.expectimax(newState, thisdepth,actor +1)
        return score/(len(gamestate.getLegalActions(actor)))

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        ghosts = gameState.getNumAgents()
        score = -1E9
        action = Directions.STOP
        for paction in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0,paction)
            tempscore= self.expectimax(newState,0, 1)
            if tempscore > score:
                score = tempscore
                action = paction
        return action
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      uses the current score as a starting point adds points if we are close enough to eat a scared ghost, or there is a opertune moment where the ghost is close to pacman, but there is enough space to grab a capsule first. Then it checks for a winning state, otherwise it will head towards a food
    """
    newPos = currentGameState.getPacmanPosition()
    newFood =currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    good_dist = 1E9
    for ghost in newGhostStates:
        cap_dist = 1E9
        dist = manhattanDistance( ghost.getPosition(), newPos)
        for cap in currentGameState.getCapsules():
            cap_dist = min(cap_dist, dist + manhattanDistance(cap, ghost.getPosition()))
        if good_dist> dist  and  dist < ghost.scaredTimer:
            dist_good =  dist
    if good_dist != 1E9:
        score += 500.0/dist_good
    if cap_dist != 1E9:
        score += 100.0/cap_dist
    if newFood.count() ==0:
        return score
    score += 10.0/manhattanDistance(newFood.asList()[0],newPos)
    return score
    "*** YOUR CODE HERE ***"
    return currentGameState.getScore()
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    def minimax(self, gamestate, thisdepth, actor, a, b):
        numActors = gamestate.getNumAgents()
        if 4 == thisdepth or gamestate.isWin() or gamestate.isLose():
            return betterEvaluationFunction(gamestate)
            
        if actor == 0:
            score = -1E9
            for paction in gamestate.getLegalActions(0):
                newState= gamestate.generateSuccessor(0,paction)
                tempscore = self.minimax(newState,thisdepth, actor +1,a,b)
                if tempscore > score:
                    score = tempscore
                if tempscore >a:
                    a = tempscore
                if score > b:
                   return score
            return score
        if actor == numActors:
            return self.minimax(gamestate,thisdepth +1, 0,a,b)
        score = 1E9
        for gaction in gamestate.getLegalActions(actor):
            newState =gamestate.generateSuccessor(actor,gaction)
            tempscore = self.minimax(newState,thisdepth, actor +1,a,b)
            if tempscore < score:
                score = tempscore
            if tempscore < b:
                b = tempscore
            if score < a:
                return score
        return score

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        ghosts = gameState.getNumAgents()
        score = -1E9
        action = Directions.STOP
        for paction in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0,paction)
            tempscore= self.minimax(newState,0, 1,score,1E9)
            if tempscore > score:
                score = tempscore
                action = paction
        return action
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

