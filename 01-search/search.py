# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from sets import Set

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    s = util.Stack() 
    state = problem.getStartState()
    bag = Set([])
    
    bag.add(state)
    for suc in problem.getSuccessors(problem.getStartState()):
        s.push((suc,[]))
        bag.add(suc[0])
    while not problem.isGoalState(state):
        next = s.pop()
        state  = next[0][0]
        lista = [] + next[1]
        lista.append(next[0][1])
        for suc in problem.getSuccessors(state):
            if problem.isGoalState(suc[0]):
               lista.append(suc[1])
               return lista 
            if suc[0] not in bag:
                s.push((suc,lista))
                bag.add(suc[0])
    return lista
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"

    node = problem.getStartState()
    if problem.isGoalState(node):
        return []
    s = util.Queue()
    s.push((node,[]))
    bag = Set([])
    fbag = set([])
    fbag.add(node)
    while 1 ==1:
        if s.isEmpty():
            return []
        node = s.pop()
        lista = [] +node[1]
        if problem.isGoalState(node[0]): 
            return lista 
        bag.add(node[0])
        for suc in problem.getSuccessors(node[0]):
            if suc[0] not in bag and suc[0] not in fbag:
                s.push((suc[0], lista +[suc[1]]))
                fbag.add(suc[0])

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "(cords, path, path cost)"

    node = problem.getStartState()
    s = util.PriorityQueue()
    s.push((node,[],0),0)
    bag = Set([])
    while 1 == 1:
        node = s.pop()
        lista = [] + node[1]
        if problem.isGoalState(node[0]):
            return lista
        for suc in problem.getSuccessors(node[0]):
            if suc[0] not in bag:
                s.push((suc[0],lista +[suc[1]],node[2] + suc[2]),node[2] +suc[2])

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    node = problem.getStartState()
    s = util.PriorityQueue()
    s.push((node,[],0),0)
    bag = Set([])
    while 1 == 1:
        node = s.pop()
        lista = [] + node[1]
        if problem.isGoalState(node[0]):
            return lista
        if node[0] in bag:
            continue
        bag.add(node[0])
        for suc in problem.getSuccessors(node[0]):
            if suc[0] not in bag:
                path_cost = node[2] + suc[2] 
                s.push((suc[0],lista +[suc[1]],path_cost),path_cost + heuristic(suc[0],problem)
)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
