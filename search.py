# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from graphicsDisplay import PacmanGraphics
from datetime import datetime
from game import Actions
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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

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

    startt= datetime.now()
    start = problem.getStartState() #start =(5,5)# in searchAgent.py
    stack = [] # DFS use stack as its structure
    explored =[]

    stack.append((start,[])) #add start node in the stack, [] is the answer

    while(True): # do a lopp
        now,ans = stack.pop()# take out the top node from the stack, and the variable now is current node and ans is path
        if problem.isGoalState(now):#test the current node if goal or not
            return ans

        if now not in explored:
            for node, direction, cost in problem.getSuccessors(now):#get the successors of node now
                history = ans + [direction]# add history path
                stack.append((node,history))#combine history path to node, if doesnt do that, when the algorithm
                                            #in the deadend and back to the previous node will cause the redundant direction
        if now not in explored:#add node to explored[]
            explored.append(now)
    over = datetime.now()
    print "time:", over-startt
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    stack = util.Queue()#BFS use queue as its struture,
    explored = []

    stack.push((start, []))#add start node in the stack, [] is the answer

    while(True):# do a lopp when stack !=0
        now, ans = stack.pop()# take out the top node from the stack, and the variable now is current node and ans is path
        #print ('now',now)
        if problem.isGoalState(now):  # test the current node is goal or not
            return ans

        if now not in explored:
            for item in problem.getSuccessors(now):

                history = ans + [item[1]]# add history path
                stack.push((item[0], history))#combine history path to node, if doesnt do that, when the algorithm
                                            #in the deadend and back to the previous node will cause the redundant direction

        if now not in explored:#add node to explored[]
            explored.append(now)




def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()  # start =(5,5)
    stack = util.PriorityQueue()#UCS use priorityqueue as a structure
    explored = []

    stack.push((start, [],0),0)#add start node in the stack, [] is the answer

    while(True):# do a lopp when stack !=0
        now, ans,total = stack.pop()# total is the totalcost of the node now
        if problem.isGoalState(now):#test the current node if goal or not
            return ans
        if now not in explored:
            for node, direction, cost in problem.getSuccessors(now):

                totalcost = total + cost
                history = ans + [direction]# add history path
                stack.push((node, history,totalcost), totalcost)#push the totalcost, so the priorityqueue can pop the smallest cost node
        if now not in explored:#add node to explored[]
            explored.append(now)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, start, goal, heuristic=nullHeuristic):
    stack = util.PriorityQueue()
    explored = []
    # base =[]
    stack.push((start, 0, [start]),
               0)  # add start node in the stack, [] is the answer, the first 0 is history cost, the second
    # one is expected cost

    while (True):  # do a lopp
        now, total, base = stack.pop()
        # print base
        # print stack
        if now == goal:  # test the current node if goal or not
            # print ans
            #print base
            return base
        if now not in explored:
            explored.append(now)
            for node, cost,direction in problem.getSuccessors(now):
                # set a if function to exclude the node have been explored
                if node not in explored:
                    totalcost = total + cost  #
                    #history = ans + [direction]  # add history path
                    abase = base + [node]
                    # print abase
                    expect = totalcost + util.manhattanDistance(goal,node)  # f(n) = g(n)+h(n):totalcost is g(n), heuristic() is h(n)
                    stack.push((node, totalcost, abase),
                               expect)  # add expected cost to the priorityQueue, and let prorityQueue according ot the expected cost to do pop()


def transfer(path):
    finalpath =[]

    for i in range(1,len(path)):
        if path[i][0] > path[i-1][0]:
            finalpath.append(Directions.EAST)
        elif path[i][0] < path[i-1][0]:
            finalpath.append(Directions.WEST)
        elif path[i][1] > path[i-1][1]:
            finalpath.append(Directions.NORTH)
        elif path[i][1] < path[i-1][1]:
            finalpath.append(Directions.SOUTH)

    return finalpath
def aStarBase(problem, heuristic=nullHeuristic):
    startt= datetime.now()
    curr = problem.getStartState()
    goal = problem.getGoalState()
    path =[]
    reach = False

    while curr!= goal and not reach:
        path_for_now = aStarSearch(problem,curr,goal)

        if path_for_now[0] == goal:
            break

        for i in range(len(path_for_now)):
            current = path_for_now[i]
            next = path_for_now[i+1]
            path.append(current)
            if problem.checkwall(next):
                curr = current
                problem.addwall(next)
                break
            elif next == goal:
                path.append(next)
                reach = True
                break
    ans = transfer(path)
    over = datetime.now()
    print "time:",over-startt

    return ans

def aStarLSearch(problem, heuristic=nullHeuristic):
    startt = datetime.now()

    queue = util.PriorityQueue()
    rhs, g = {}, {}
    goal, start = problem.getGoalState(), problem.getStartState()
    path = []
    reach = False
    def calculateKey(s):
        return min(g[s],rhs[s])+util.manhattanDistance(s,goal),min(g[s],rhs[s])

    def initilize():
        for state in problem.allstates():
            rhs[state] = g[state] = float('inf')
        rhs[start] = 0
        queue.push(start,calculateKey(start))

    def updateVertex(u):
        if u != start:
            next_states = problem.getSuccessors(u)

            rhs[u] = min((g[state]+cost) for state,cost,direction in next_states)
        if u in queue.contain:
            queue.remove(u)
        if g[u]!=rhs[u]:
            queue.push(u,calculateKey(u))

    def computeShortestPath():
        while queue.topKey() < calculateKey(goal) or rhs[goal] != g[goal]:

            u  = queue.pop()
            preu = problem.getSuccessors(u)

            if g[u]>rhs[u]:
                g[u] = rhs[u]

                for successor ,cost,action in preu:
                    updateVertex(successor)
            else:
                g[u] = float('inf')
                preu.append((u,1,None))
                for successor,cost,directions in preu:
                    updateVertex(successor)
    def printg():
        for i in problem.allstates():
            print i,g[i]

    def findpath(start):
        tmppath = []

        goal =problem.getGoalState()
        #print start
        explored = [goal]
        while goal != start:
            temp = float('inf')

            tmppath.append(goal)

            for successor, cost, direction in problem.getSuccessors(goal):
                #print g[successor]
                if temp > g[successor] and successor not in explored:
                    temp = g[successor]

                    nextstate =successor
            explored.append(nextstate)
            goal=nextstate

        tmppath.append(start)
        return tmppath[::-1]


    current = start
    while not reach:# same to astarbase search
        initilize()

        computeShortestPath()
        #print len(path),path
        path_for_now = findpath(current)# like astarsearch, find the closest path to goal
        #print path_for_now
        #printg()
        #break


        rangeRoute = range(len(path_for_now))
        for i in rangeRoute:
            current = path_for_now[i]
            next = path_for_now[i + 1]
            path.append(path_for_now[i])
            if problem.checkwall(next) == True:
                start = current
                problem.addwall(next)
                updateVertex(next)
                break
            elif next is goal:
                path.append(goal)
                reach = True
                break

    over = datetime.now()
    print "time:",over-startt
    return transfer(path)


def dStarSearch(problem, heuristic=nullHeuristic):
    startt = datetime.now()
    km =0
    Pqueue = util.PriorityQueue()
    rhs, g = {}, {}

    start, goal = problem.getStartState(), problem.getGoalState()
    path = []



    def calculateKey(s):
        k1 = min(g[s],rhs[s]) + util.manhattanDistance(start,s) + km
        k2 = min(g[s],rhs[s])
        return (k1,k2)
    def initilize():
        while not Pqueue.isEmpty():
            Pqueue.pop()
        for state in problem.allstates():
            rhs[state] = g[state] = float('inf') # set the every nodes in map are infinite
        rhs[goal] = 0 # set the rhs value of goal 0
        Pqueue.push(goal,(util.manhattanDistance(start,goal),0))# insert the goal nad its key value

    def updateVertex(u):

        """if g[u] != rhs[u] and u in Pqueue.contain:
            Pqueue.update(u, calculateKey(u))
        elif g[u] != rhs[u] and u not in Pqueue.contain:
            Pqueue.push(u, calculateKey(u))
        elif g[u] == rhs[u]and u in Pqueue.contain:
            Pqueue.remove(u)"""
        if u!=goal: # if the input is not the goal, let the rhs value of input euqal to the min value of its successor's rhs value
            rhs[u] = min((g[state]+cost) for state,cost,_ in problem.getSuccessors(u))

        if u in Pqueue.contain:# if u in the Priorityqueue, remove it
            Pqueue.remove(u)
        if g[u] != rhs[u]:
            Pqueue.push(u, calculateKey(u))
    def printg():
        for i in problem.allstates():
            print i,g[i]


    def computeShortestPath():

        while Pqueue.topKey() < calculateKey(start) or rhs[start] > g[start]:
            kold = Pqueue.topKey()
            u = Pqueue.pop()
            #print "u:",u
            #print pre
            knew = calculateKey(u)
            if kold < knew:

                Pqueue.update(u,knew)
            elif g[u] > rhs[u]:
                g[u] = rhs[u]
                Pqueue.remove(u)

                for successor, cost,_ in problem.getSuccessors(u):
                    if successor !=goal:
                        rhs[successor] = min(rhs[successor],cost+g[u])
                    updateVertex(successor)
            else:
                gold = g[u]
                g[u] = float('inf')#encounter the walls
                preu = problem.getSuccessors(u)
                preu.append((u, 1,None))
                #minn = float('inf')
                for successor, cost, _ in preu:#updatet the presucessor of node u
                    #if rhs[successor] == cost+gold:
                    #    if successor != goal:
                    #       rhs[successor] = min((cost1+g[state]) for state,cost1,_ in problem.getSuccessors(successor))


                    #updateVertex(successor)
                    updateVertex(successor)

    slast = start
    initilize()
    computeShortestPath()
    #printg()

    while start != goal:
        mini = float('inf')
        ministate = start

        for successor, cost,direction in problem.getSuccessors(start):

            temp = g[successor]+cost
            if temp<mini:
                mini = temp
                ministate = successor
        cold = mini
        #print mini
        if problem.checkwall(ministate) == True: # check whether there is a walls, if not move the ministate
            problem.addwall(ministate)
            km += util.manhattanDistance(slast,start)
            slast = start
            #if start != goal and cold > 1:
            #    rhs[start] = min((cost + g[state]) for state, cost, direction in problem.getSuccessors(ministate))
            #updateVertex(start)

        else:
            #updateVertex(ministate)
            """for successor,cost,direction in problem.getSuccessors(start):
                if successor == ministate:
                    newc = cost
                    #dx,dy = Actions.directionToVector(direction)
            u = start
            v = ministate
            if cold > newc:
                if u!=goal:
                    #print rhs[start],newc,g[ministate]
                    rhs[u] = min(rhs[u],newc+g[v])
            elif rhs[u] == cold+g[v]:
                if u !=goal:
                    rhs[u] = min((cost+g[state]) for state,cost,direction in problem.getSuccessors(ministate))
            #v = (successor[0] + dx,successor[1]+dy)"""

            """if (cold > newc):
                if u!=goal:
                    rhs[u] = min(rhs[u],newc + g[v])
            elif u != goal and rhs[u] == cold+g[v]:

                rhs[ministate] = min((cost + g[state]) for state, cost, direction in problem.getSuccessors(ministate))
            #updateVertex(ministate)"""
            #updateVertex(start)
            path.append(start)
            start = ministate
        updateVertex(start)
        computeShortestPath()
    path.append(goal)
    directions = transfer(path)
    stop = datetime.now()
    elapsed_time = stop - startt
    print elapsed_time

    return directions




bfs = breadthFirstSearch
dfs = depthFirstSearch
astarb = aStarBase
astar = aStarSearch
lstar = aStarLSearch
ucs = uniformCostSearch
dstar = dStarSearch
