# THIS IS THE ONE WE TURNED IN

# MinMax AI, HW 3 CS-421
# Authors:
# - Makengo Lokombo (lokombo27)
# - Joshua Krasnogorov (krasnogo27)
# - Chengen Li (lic27)

import random
import sys

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
# rootEval
#
# Evaluates the root node, base case for miniMax
#
# Parameters:
#   gameState - a game state
#   me - the id of me, the current player
#
# Return: The utility of the state
#
def rootEval(gameState, me):
    if gameState.whoseTurn == me:
        return utility(gameState)
    else:
        return -utility(gameState)


##
# miniMax
#
# Runs the minimax algorithm on a given node with alpha-beta pruning
#
# Parameters:
#   gameState - a game state
#   depth - the depth of the search
#   alpha - the alpha value for alpha-beta pruning
#   beta - the beta value for alpha-beta pruning
#   me - the id of me, the current player
#
# Return: A tuple containing (bestValue, bestMove)
#
def miniMax(gameState, depth, alpha, beta, me):
    if gameState.phase == PLAY_PHASE:
        winner = getWinner(gameState)
        if winner is not None or depth == 0:  # base case
            return rootEval(gameState, me), None

    moves = listAllLegalMoves(gameState)

    if not moves:
        return rootEval(gameState, me), None
    
    # Sort moves for better pruning
    if gameState.whoseTurn == me:
        # Sort descending (best first)
        moves.sort(key=lambda move: rootEval(
            getNextStateAdversarial(gameState, move), me), reverse=True)
    else:
        # Sort ascending (worst first)
        moves.sort(key=lambda move: rootEval(
            getNextStateAdversarial(gameState, move), me))
    
    # Limit to top 12 moves for better performance
    moves = moves[:12]
    
    # If it's my turn, we want to maximize our score
    if gameState.whoseTurn == me:
        bestValue = float('-inf')
        bestMove = None
        for move in moves:
            newState = getNextStateAdversarial(gameState, move)
            value, _ = miniMax(newState, depth - 1, alpha, beta, me)
            if value > bestValue:
                bestValue = value
                bestMove = move

            # update alpha
            alpha = max(alpha, bestValue)
            # prune if beta <= alpha
            if beta <= alpha:
                break

        return bestValue, bestMove
    # If it's the enemy's turn, we want to minimize our score
    else:
        bestValue = float('inf')
        bestMove = None
        for move in moves:
            newState = getNextStateAdversarial(gameState, move)
            value, _ = miniMax(newState, depth - 1, alpha, beta, me)

            if value < bestValue:
                bestValue = value
                bestMove = move

            # update beta
            beta = min(beta, bestValue)
            # prune if beta <= alpha
            if beta <= alpha:
                break

        return bestValue, bestMove


##
# utility
#
# Calculates the utility of a given game state on a scale of 0 to 1
# Reminder: Do not use the board variable
#
# Parameters:
#   gameState - a game state
#
# Return: The utility of the state
#
def utility(gameState):
    # Constants
    me = gameState.whoseTurn
    enemy = 1 - me
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(enemy, gameState)
    myAnts = getAntList(gameState, me, (WORKER, DRONE, SOLDIER, R_SOLDIER, QUEEN))
    utility = 0.0
    
    # Check for instant win/lose conditions
    if gameState.phase == PLAY_PHASE:
        if (getWinner(gameState) == me or 
                len(getAntList(gameState, enemy, (QUEEN,))) == 0 or
                myInv.foodCount == FOOD_GOAL or
                enemyInv.getAnthill().captureHealth == 0):
            return 100.0  # instant win
        elif (getWinner(gameState) == enemy or
              len(myAnts) == 0 or
              enemyInv.foodCount == FOOD_GOAL or
              myInv.getAnthill().captureHealth == 0):
            return 0.0  # instant lose

    # Calculate utility components
    foodScore = foodUtility(gameState, myInv, enemyInv, me)
    defenseScore = defenseUtility(gameState, me)
    attackScore = attackUtility(gameState, myInv, enemyInv, me)
    
    utility = foodScore + defenseScore + attackScore
    return utility


##
# foodUtility
#
# Calculates the utility of the food situation in a game state
# Includes worker utility
#
# Parameters:
#   gameState - a game state
#   myInv - the inventory of the current player
#   enemyInv - the inventory of the enemy
#   me - the id of the current player
#
# Return: The utility of the food situation
#
def foodUtility(gameState, myInv, enemyInv, me):
    utility = 0.0
    # Food Weights - 90% of total utility
    if myInv.foodCount is not None and enemyInv.foodCount is not None:
        foodScore = 0.5
        foodScore += (myInv.foodCount / FOOD_GOAL) * 0.5
        foodScore -= (enemyInv.foodCount / FOOD_GOAL) * 0.5
        utility += foodScore * 0.95

        # Worker utility calculation
        workerScore = 0.0
        # Get my workers
        myWorkers = getAntList(gameState, me, (WORKER,))
        tunnels = myInv.getTunnels()
        anthill = myInv.getAnthill()
        foodList = getConstrList(gameState, None, (FOOD,))
        foodList = [f for f in foodList if (f.coords[1] <= 4)]
        numWorkers = len(myWorkers)

        # If we have no workers, score is 0
        if numWorkers == 0:
            return 0.0
        
        # Penalty for too many workers
        for i in range(len(myWorkers)):
            if i > 2:
                utility -= 0.1

        # Calculate worker contributions
        if numWorkers > 0:
            # Precompute drop sites
            dropSites = []
            if anthill:
                dropSites.append(anthill.coords)
            if tunnels:
                dropSites.extend([t.coords for t in tunnels])

            # Normalization constants keep per-worker contribution in [0,1]
            MAX_FOOD_DIST = 12.0
            MAX_DROP_DIST = 12.0

            for i, w in enumerate(myWorkers):
                contrib = 0.0

                if w.carrying:
                    # If at drop site: full contribution
                    if dropSites and any(w.coords == d for d in dropSites):
                        contrib = 1.0
                    else:
                        # Positive baseline for carrying so picking up is attractive
                        if dropSites:
                            closestDrop = min(approxDist(w.coords, d) for d in dropSites)
                            progressToDrop = max(0.0, min(1.0, 
                                1.0 - (closestDrop / MAX_DROP_DIST)))
                        else:
                            progressToDrop = 0.0
                        # Baseline 0.5 plus progress up to 1.0 max
                        contrib = 0.5 + 0.5 * progressToDrop
                else:
                    # Not carrying: incentivize getting closer to nearest food
                    if foodList:
                        closestFood = min(approxDist(w.coords, f.coords) 
                                        for f in foodList)
                        towardFood = max(0.0, min(1.0, 
                            1.0 - (closestFood / MAX_FOOD_DIST)))
                        contrib = 0.5 * towardFood
                    else:
                        contrib = 0.0

                # Clamp and average across workers
                contrib = max(0.0, min(1.0, contrib))
                workerScore += contrib / numWorkers

        # Ensure workerScore in [0,1]
        workerScore = max(0.0, min(1.0, workerScore))
        utility += (workerScore * 0.03)
        
    return utility


##
# defenseUtility
#
# Calculates the utility of the defense situation in a game state
#
# Parameters:
#   gameState - a game state
#   me - the id of the current player
#
# Return: The utility of the defense situation
#
def defenseUtility(gameState, me):
    enemy = 1 - me
    
    def on_my_side(coords):
        y = coords[1]
        return (y <= 4)

    # Enemy ants on my side are threats
    threats = [a for a in getAntList(gameState, enemy, 
        (WORKER, DRONE, SOLDIER, R_SOLDIER)) if on_my_side(a.coords)]
    # My attack-capable ants
    defenders = getAntList(gameState, me, (DRONE,))

    # If no threats on my side, defense is perfect
    if not threats:
        return 1.0
    # If there are threats but no defenders, defense is bad
    if not defenders:
        return 0.0

    # Encourage defenders to be close to threats
    # 0 distance -> 1.0 score; distance >= maxDist -> 0.0 score
    MAX_DIST = 14.0
    total = 0.0
    for d in defenders:
        minDist = min(approxDist(d.coords, t.coords) for t in threats)
        score = 1.0 - (minDist / MAX_DIST)
        total += score

    proximityScore = total / len(defenders)
    return proximityScore


##
# attackUtility
#
# Calculates the utility of the attack situation in a game state
#
# Parameters:
#   gameState - a game state
#   myInv - the inventory of the current player
#   enemyInv - the inventory of the enemy
#   me - the id of the current player
#
# Return: The utility of the attack situation
#
def attackUtility(gameState, myInv, enemyInv, me):
    enemy = 1 - me
    MAX_DIST = 18.0

    # Helper function to calculate proximity score
    def proximityScore(ants, enemyAnts):
        total = 0.0
        for ant in ants:
            if enemyAnts:
                minDist = min(approxDist(ant.coords, enemyAnt.coords) 
                            for enemyAnt in enemyAnts)
                score = 1.0 - (minDist / MAX_DIST)
                total += score
        return total / len(ants)
 
    # Only consider drones for attack utility
    myAnts = getAntList(gameState, me, (DRONE,))
    if not myAnts:
        return 0.0
    
    # Get enemy workers as primary targets
    enemyWorkers = getAntList(gameState, enemy, (WORKER,))

    # Prioritize attacking workers
    if enemyWorkers:
        workerProximityScore = proximityScore(myAnts, enemyWorkers)
        # punish for more workers
        for w in enemyWorkers:
            workerProximityScore -= 0.05

        return workerProximityScore
    
    # get distance from top right corner of board, limits the amount of moves drone can make
    TOP_RIGHT_CORNER = (0, 9)
    topRightCornerScore = 1.0 - (approxDist(myAnts[0].coords, TOP_RIGHT_CORNER) / MAX_DIST)
    return 2.0 + topRightCornerScore  # no more workers, great


##
# AIPlayer
#
# The responsibility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
#
class AIPlayer(Player):

    ##
    # __init__
    #
    # Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Genetic Algorithm Bot")
        self.playerId = inputPlayerId


    ##
    # getPlacement
    #
    # Called during setup phase for each Construction that
    # must be placed by the player. These items are: 1 Anthill on
    # the player's side; 1 tunnel on player's side; 9 grass on the
    # player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    #
    def getPlacement(self, currentState):
        numToPlace = 0
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move is None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if (currentState.board[x][y].constr is None and 
                            (x, y) not in moves):
                        move = (x, y)
                        # Just need to make the space non-empty
                        currentState.board[x][y].constr = True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move is None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if (currentState.board[x][y].constr is None and 
                            (x, y) not in moves):
                        move = (x, y)
                        # Just need to make the space non-empty
                        currentState.board[x][y].constr = True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    

    ##
    # getMove
    #
    # Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    #
    def getMove(self, currentState):
        # run miniMax
        _, move = miniMax(currentState, 3, float('-inf'), float('inf'), self.playerId)
        return move


    ##
    # getAttack
    #
    # Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocations - The Locations of the Enemies that can be attacked (Location[])
    #
    # Return: The location to attack
    #
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # registerWin
    #
    # This agent doesn't learn
    #
    # Parameters:
    #   hasWon - whether the agent won the game
    #
    def registerWin(self, hasWon):
        # method template, not implemented
        pass


# Unit tests for MinMax AI
TOTAL_TESTS = 5
passedTests = 0

# UTILITY TEST
try:
    gameState = GameState.getBlankState()
    util = utility(gameState)
    if not -10.0 <= util <= 100.0:
        print(f"ERROR: utility() returned {util}, expected -10.0-100.0")
    else:
        passedTests += 1
except Exception as e:
    print(f"ERROR: utility() test failed with exception: {e}")


# FOOD UTILITY TEST
try:
    gameState = GameState.getBasicState()
    util = foodUtility(gameState, getCurrPlayerInventory(gameState), 
                      getEnemyInv(0, gameState), 0)
    if not -1.0 <= util <= 1.0:
        print(f"ERROR: foodUtility() returned {util}, expected -1.0-1.0")
    else:
        passedTests += 1
except Exception as e:
    print(f"ERROR: foodUtility() test failed with exception: {e}")


# DEFENSE UTILITY TEST
try:
    gameState = GameState.getBasicState()
    util = defenseUtility(gameState, 0)
    if not 0.0 <= util <= 1.0:
        print(f"ERROR: defenseUtility() returned {util}, expected 0.0-1.0")
    else:
        passedTests += 1
except Exception as e:
    print(f"ERROR: defenseUtility() test failed with exception: {e}")


# ATTACK UTILITY TEST
try:
    gameState = GameState.getBasicState()
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(0, gameState)
    util = attackUtility(gameState, myInv, enemyInv, 0)
    if not 0.0 <= util <= 3.0:
        print(f"ERROR: attackUtility() returned {util}, expected 0.0-3.0")
    else:
        passedTests += 1
except Exception as e:
    print(f"ERROR: attackUtility() test failed with exception: {e}")
