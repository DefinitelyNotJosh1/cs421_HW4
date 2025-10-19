# THIS IS THE ONE WE TURNED IN

# MinMax AI, HW 3 CS-421
# Authors:
# - Makengo Lokombo (lokombo27)
# - Joshua Krasnogorov (krasnogo27)
# - Chengen Li (lic27)

import random
import sys
import os

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
#   util_fn - the utility function to use (will be in AI player class)
#
# Return: The utility of the state
#
def rootEval(gameState, me, util_fn):
    if gameState.whoseTurn == me:
        return util_fn(gameState, me)
    else:
        return -util_fn(gameState, me)


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
#   util_fn - the utility function to use (will be in AI player class)
#
# Return: A tuple containing (bestValue, bestMove)
#
def miniMax(gameState, depth, alpha, beta, me, util_fn):
    if gameState.phase == PLAY_PHASE:
        winner = getWinner(gameState)
        if winner is not None or depth == 0:  # base case
            return rootEval(gameState, me, util_fn), None

    moves = listAllLegalMoves(gameState)

    if not moves:
        return rootEval(gameState, me, util_fn), None
    
    # Sort moves for better pruning
    if gameState.whoseTurn == me:
        # Sort descending (best first)
        moves.sort(key=lambda move: rootEval(
            getNextStateAdversarial(gameState, move), me, util_fn), reverse=True)
    else:
        # Sort ascending (worst first)
        moves.sort(key=lambda move: rootEval(
            getNextStateAdversarial(gameState, move), me, util_fn))
    
    # Limit to top 12 moves for better performance
    moves = moves[:12]
    
    # If it's my turn, we want to maximize our score
    if gameState.whoseTurn == me:
        bestValue = float('-inf')
        bestMoves = []
        for move in moves:
            newState = getNextStateAdversarial(gameState, move)
            value, _ = miniMax(newState, depth - 1, alpha, beta, me, util_fn)
            if value > bestValue:
                bestValue = value
                bestMoves = [move]
            elif abs(value - bestValue) < 0.000001:
                bestMoves.append(move)

            # update alpha
            alpha = max(alpha, bestValue)
            # prune if beta <= alpha
            if beta <= alpha:
                break
        
        move = random.choice(bestMoves) if bestMoves else None
        return bestValue, move
    # If it's the enemy's turn, we want to minimize our score
    else:
        bestValue = float('inf')
        bestMoves = []
        for move in moves:
            newState = getNextStateAdversarial(gameState, move)
            value, _ = miniMax(newState, depth - 1, alpha, beta, me, util_fn)

            if value < bestValue:
                bestValue = value
                bestMoves = [move]
            elif abs(value - bestValue) < 0.000001:
                bestMoves.append(move)

            # update beta
            beta = min(beta, bestValue)
            # prune if beta <= alpha
            if beta <= alpha:
                break
        
        # Return random move from best moves
        move = random.choice(bestMoves) if bestMoves else None
        return bestValue, move

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
        # Setup instance variables (pretty self-explanatory)
        self.genePop = []
        self.nextGene = 0
        self.fitness = []
        self.evalCounts = []
        self.populationSize = 10
        self.evalGames = 5
        self.populationFile = os.path.join(os.path.dirname(__file__), '..', 'vo27_krasnogo27_population.txt')
        self.featureCount = 17
        self.mutationRate = 0.03 # 3% chance of mutation (may change we'll see)
        self.initializePopulation()


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
        _, move = miniMax(currentState, 3, float('-inf'), float('inf'), self.playerId, self.utility)
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
        # If no population, return
        if not self.genePop:
            return
        
        # Keep index in bounds
        index = self.nextGene
        if index >= self.populationSize:
            return

        # Update the fitness scores
        self.fitness[index] += 1.0 if hasWon else 0.0
        self.evalCounts[index] += 1
        # Judge whether current gene fitness has been fully evaluated by checking (N games evaluated). If yes, go to next gene
        if self.evalCounts[self.nextGene] == self.evalGames:
            self.nextGene += 1
            # If all genes have been evaluated, generate a new population
            if self.nextGene == self.populationSize:
                self.nextGeneration()
                self.nextGene = 0


    ##########################
    # Genetic Algorithm Methods
    ##########################

    ## 
    # initializePopulation
    #
    # Initializes the population of genes
    #
    # Return: Saves the population of genes
    #
    def initializePopulation(self):
        # load the population from the file if it exists
        loaded = []
        # try to load the population from the file
        try:
            if os.path.exists(self.populationFile):
                with open(self.populationFile, "r") as f:
                    # read the file line by line
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p.strip() for p in line.replace(" ", "").split(",")]
                        row = [self._clamp(float(p)) for p in parts if p != ""]
                        if len(row) == self.featureCount:
                            loaded.append(row)
            
            # If file content doesn't match pop size, regenerate - avoids issues in the future if we change the population size
            if len(loaded) != self.populationSize:
                loaded = [[random.uniform(-10.0, 10.0) for _ in range(self.featureCount)] for __ in range(self.populationSize)]
        except Exception:
            # On any error, regenerate
            loaded = [[random.uniform(-10.0, 10.0) for _ in range(self.featureCount)] for __ in range(self.populationSize)]
        self.genePop = loaded
        self.fitness = [0.0 for _ in range(self.populationSize)]
        self.evalCounts = [0 for _ in range(self.populationSize)]
        self.nextGene = 0
        # save the population to the file
        self.savePopulation()

        ## 
        # savePopulation
        #
        # saves the population of genes to the population file
        #
    def savePopulation(self):
        with open(self.populationFile, 'w') as f:
            for gene in self.genePop:
                # write features to file, 6 decimal places
                f.write(','.join(f"{g:.6f}" for g in gene) + '\n')
    
    ##
    # mate
    # 
    # takes two parent genes and mates them to create two child genes
    #
    # Parameters:
    #   parent1 - the first parent gene
    #   parent2 - the second parent gene
    #
    # Return: Two child genes
    #
    def mate(self, parent1, parent2):
        # mate the genes 
        # cut the genes at a random point in the middle
        cut = random.randint(1, self.featureCount - 1)
        child1 = parent1[:cut] + parent2[cut:]
        child2 = parent2[:cut] + parent1[cut:]

        # mutate the children
        for i in range(len(child1)):
            if random.random() < self.mutationRate:
                child1[i] = random.uniform(-10, 10)
        for i in range(len(child2)):
            if random.random() < self.mutationRate:
                child2[i] = random.uniform(-10, 10)
        
        return child1, child2


    ##
    # nextGeneration
    #
    # generates the next generation of genes and saves them to the population file
    #
    # Return: None
    def nextGeneration(self):
        # Select the parents for the next generation
        parents = sorted(zip(self.fitness, self.genePop), key=lambda x: x[0])
        parents = parents[self.populationSize // 2:]

        # Mate the parents to create the next generation
        nextGeneration = []
        pairsNeeded = self.populationSize // 2
        for i in range(pairsNeeded):
            p1 = parents[i % len(parents)][1]
            p2 = parents[(i + 1) % len(parents)][1]
            c1, c2 = self.mate(p1, p2)
            nextGeneration.append(c1)
            nextGeneration.append(c2)

        # Save the next generation to the population file
        self.genePop = nextGeneration
        self.savePopulation()

        # Reset the evaluation counts
        self.evalCounts = [0 for _ in range(self.populationSize)]
        # Reset the next gene
        self.nextGene = 0
        # Reset the fitness scores
        self.fitness = [0.0 for _ in range(self.populationSize)]

    ##
    # utility
    #
    # Calculates the utility of a given game state. Calculates the features and returns the sum of features
    #
    # Parameters:
    #   gameState - a game state
    #
    # Return: The utility of the state
    #
    def utility(self, gameState, me):
        # Constants
        enemy = 1 - me
        myInv = getCurrPlayerInventory(gameState)
        enInv = getEnemyInv(enemy, gameState)
        
        # My values
        myWorkers = getAntList(gameState, me, (WORKER,))
        myDrones = getAntList(gameState, me, (DRONE,))
        mySoldiers = getAntList(gameState, me, (SOLDIER,))
        myRangedSoldiers = getAntList(gameState, me, (R_SOLDIER,))
        myQueen = getAntList(gameState, me, (QUEEN,))[0] if getAntList(gameState, me, (QUEEN,)) else None
        myOff = myDrones + mySoldiers + myRangedSoldiers
        myHill = myInv.getAnthill()
        myTunnels = myInv.getTunnels()

        # Enemy values
        enWorkers = getAntList(gameState, enemy, (WORKER,))
        enDrones = getAntList(gameState, enemy, (DRONE,))
        enSoldiers = getAntList(gameState, enemy, (SOLDIER,))
        enRangedSoldiers = getAntList(gameState, enemy, (R_SOLDIER,))
        enQueen = getAntList(gameState, enemy, (QUEEN,))[0] if getAntList(gameState, enemy, (QUEEN,)) else None
        enOff = enDrones + enSoldiers + enRangedSoldiers
        enHill = enInv.getAnthill()

        # Food values
        foods = getConstrList(gameState, None, (FOOD,))

        # Helper functions

        ##
        # avgDist
        #
        # Finds average distance from a list of ants to a target
        #
        # Parameters:
        #   ants - a list of ants
        #   target - the target location
        #
        # Return: The average distance
        #
        def avgDist(ants, target):
            if not ants or target is None:
                return 0.0
            return sum(approxDist(ant.coords, target) for ant in ants) / len(ants)

        ##
        # avgDistList
        #
        # Finds average distance from a list of ants to a list of targets
        #
        # Parameters:
        #   ants - a list of ants
        #   targets - a list of targets
        #
        # Return: The average distance
        #
        def avgDistList(ants, targets):
            if not ants or not targets:
                return 0.0
            return sum(avgDist(ants, target) for target in targets) / len(targets)
        
        # Features
        f1 = myInv.foodCount - enInv.foodCount                                  # my food count - enemy food count
        f2 = myQueen.health - enQueen.health if myQueen and enQueen else 0      # my queen health - enemy queen health
        f3 = len(myDrones) - len(enDrones)                                      # my drone count - enemy drone count
        f4 = len(mySoldiers) - len(enSoldiers)                                  # my soldier count - enemy soldier count
        f5 = len(myWorkers) - len(enWorkers)                                    # my worker count - enemy worker count
        f6 = len(myRangedSoldiers) - len(enRangedSoldiers)                      # my ranged soldier count - enemy ranged soldier count
        f7 = avgDist(myOff, enQueen if enQueen else None)                       # average distance from my offensive ants to enemy queen
        f8 = avgDist(enOff, myQueen.coords if myQueen else None)                # average distance from enemy offensive ants to my queen
        f9 = 1 if len(myOff) > len(enOff) else 0                                # 1 if I have more offensive ants than my opponent, 0 if I don't
        f10 = avgDist(myOff, enHill if enHill else None)                        # average distance from my offensive ants to enemy anthill
        f11 = avgDist(enOff, myHill if myHill else None)                        # average distance from enemy offensive ants to my anthill
        f12 = 0
        f13 = 0
        numCarrying = 0
        numEmpty = 0
        for w in myWorkers:
            if w.carrying:
                numCarrying += 1
                f12 += min(approxDist(w.coords, myHill.coords), approxDist(w.coords, myTunnels[0].coords))
            else:
                numEmpty += 1
                f13 += min(approxDist(w.coords, food.coords) for food in foods)
        f12 = f12 / numCarrying if numCarrying else 0                                     # average distance from workers with food from tunnel or hill (whichever is closer)
        f13 = f13 / numEmpty if numEmpty else 0                                           # average distance from workers without food from food source (closest food source)
        f14 = avgDistList(myWorkers, enOff) if myWorkers and enOff else 0                 # average distance from my workers to enemy offensive ants
        f15 = avgDistList(enWorkers, myOff) if enWorkers and myOff else 0                 # average distance from enemy workers to my offensive ants
        f16 = approxDist(myQueen.coords, enQueen.coords) if myQueen and enQueen else 0    # distance from my queen to enemy queen
        f17 = avgDistList(mySoldiers, enSoldiers) if mySoldiers and enSoldiers else 0     # average distance from my soldiers to enemy soldiers

        # return the sum of the features
        return f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17




# UNIT TESTS

# Test initialize Population
