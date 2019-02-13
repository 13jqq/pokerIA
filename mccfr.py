import numpy as np
import config

class Node():

    def __init__(self, state):
        self.state = state
        self.playerTurn = state.playerTurn
        self.id = state.id
        self.edges = []

    def isLeaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True

class Edge():

    def __init__(self, inNode, outNode, prior, action):
        self.id = inNode.state.id + '|' + outNode.state.id
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.state.playerTurn
        self.action = action

        self.stats = {
            'N': 0,
            'R': 0,
            'P': prior
        }

class MCCFR():

    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.addNode(root)

    def __len__(self):
        return len(self.tree)

    def moveToLeaf(self):

        breadcrumbs = []
        currentNode = self.root

        done = 0
        value = {}

        while not currentNode.isLeaf():
            maxRU = float('-inf')

            if currentNode == self.root:
                epsilon = config.mccfr['EPSILON']
                nu = np.random.dirichlet([config.mccfr['ALPHA']] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):

                U = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Nb) / (1 + edge.stats['N'])

                R = edge.stats['R']

                if R + U > maxRU:
                    maxRU = R + U
                    simulationAction = action
                    simulationEdge = edge

            newState, value, done = currentNode.state.takeAction(
                simulationAction)  # the value of the newState from the POV of the new playerTurn
            breadcrumbs.append((currentNode,simulationEdge))
            currentNode = simulationEdge.outNode

        return currentNode, value, done, breadcrumbs

    def backFill(self, value, breadcrumbs):

        for idx, (node,selEdge) in enumerate(breadcrumbs):
            pisigziaz = np.prod([e[1].stats['P'] for e in breadcrumbs[idx + 1:]])
            pisigziz = np.prod([e[1].stats['P'] for e in breadcrumbs[idx:]])
            piadvzi = np.prod([e[1].stats['P'] for e in breadcrumbs[:idx] if e[0].playerTurn != node.playerTurn])
            pisigprimez = 1  # The sample of terminal history is deterministic

            for action, edge in node.edges:
                if selEdge.id == edge.id:
                    edge.stats['N'] = edge.stats['N'] + 1
                    edge.stats['R'] = edge.stats['R'] + ((value[node.playerTurn] * piadvzi)/pisigprimez) * (pisigziaz - pisigziz)
                else:
                    edge.stats['R'] = edge.stats['R'] - ((value[node.playerTurn] * piadvzi)/pisigprimez) * pisigziz

    def addNode(self, node):
        self.tree[node.id] = node

