from utilities import stable_softmax
import numpy as np
import config
import math

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

    def __init__(self, root, my_uuid):
        self.my_uuid = my_uuid
        self.root = root
        self.tree = {}
        self.addNode(root)

    def __len__(self):
        return len(self.tree)

    def moveToLeaf(self, model):

        breadcrumbs = []
        currentNode = self.root

        done = 0
        value = 0

        while not currentNode.isLeaf():

            #We don't need this value anymore
            #maxRU = float('-inf')

            if currentNode == self.root:
                epsilon = config.mccfr['EPSILON']
                nu = np.random.gamma(config.mccfr['ALPHA'], 1, len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            Nb = np.sum([edge.stats['N'] for action, edge in currentNode.edges])

            # Replacing for loop with list completion and np sum func for speeding code
            #for action, edge in currentNode.edges:
            #    Nb = Nb + edge.stats['N']

            if currentNode.playerTurn == self.my_uuid:

                idx = np.argmax([((math.log((config.mccfr['CPUCT_BASE'] + 1 + edge.stats['N'])/config.mccfr['CPUCT_BASE']) + config.mccfr['CPUCT_INIT']) * \
                        ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(Nb) / (1 + edge.stats['N'])) + edge.stats['R'] for (action, edge) in currentNode.edges])

                simulationAction, simulationEdge = currentNode.edges[idx]

                #Replacing for loop with list completion and np argmax func for speeding code
                #for idx, (action, edge) in enumerate(currentNode.edges):

                #    U = ((math.log((config.mccfr['CPUCT_BASE'] + 1 + edge.stats['N'])/config.mccfr['CPUCT_BASE']) + config.mccfr['CPUCT_INIT']) * \
                #        ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                #        np.sqrt(Nb) / (1 + edge.stats['N']))

                #    R = edge.stats['R']

                #    if R + U > maxRU:
                #        maxRU = R + U
                #        simulationAction = action
                #        simulationEdge = edge

            else:
                main_input, my_info, my_history, adv_info, adv_history = currentNode.state.convertStateToModelInput()
                preds = model.predict([main_input, my_info, my_history, *adv_info, *adv_history])
                logits_array = preds[1]
                logits = logits_array[0]
                allowedActions = currentNode.state.allowed_action
                mask = np.ones(logits.shape, dtype=bool)
                mask[allowedActions] = False
                logits[mask] = float('-inf')
                probs = stable_softmax(logits)
                probs = probs[allowedActions]
                action_idx = np.random.multinomial(1, probs)
                chosen_action = np.where(action_idx == 1)[0][0]

                # With the simplification in the backfill function to calculate regret, we don't need the adv prob values anymore
                #for idx, (action, edge) in enumerate(currentNode.edges):
                #    edge.stats['P'] = probs[idx]

                simulationAction = currentNode.edges[chosen_action][0]
                simulationEdge = currentNode.edges[chosen_action][1]

            newState, value, done = currentNode.state.takeAction(
                simulationAction)  # the value of the newState from the POV of the new playerTurn
            breadcrumbs.append((currentNode, simulationEdge))
            currentNode = simulationEdge.outNode

        return currentNode, value, done, breadcrumbs

    def backFill(self, value, breadcrumbs):

        for idx, (node,selEdge) in enumerate(breadcrumbs):
            if node.playerTurn == self.my_uuid:
                simplcoeff = np.prod([e[1].stats['P'] for e in breadcrumbs[idx + 1:] if e[0].playerTurn == node.playerTurn])

                #Simplification of all the values underneath with simplcoeff
                #pisigziaz = np.prod([e[1].stats['P'] for e in breadcrumbs[idx + 1:]])
                #pisigziz = np.prod([e[1].stats['P'] for e in breadcrumbs[idx:]])
                #piadvzi = np.prod([e[1].stats['P'] for e in breadcrumbs[:idx] if e[0].playerTurn != node.playerTurn])
                #pisigprimez = 1 * np.prod([e[1].stats['P'] for e in breadcrumbs if e[0].playerTurn != node.playerTurn])


                for action, edge in node.edges:
                    if selEdge.id == edge.id:
                        edge.stats['N'] = edge.stats['N'] + 1

                        #Rewriting the new simplificated regret expression with simplecoeff
                        #edge.stats['R'] = edge.stats['R'] + ((value * piadvzi)/pisigprimez) * (pisigziaz - pisigziz)

                        edge.stats['R'] = edge.stats['R'] + (value * simplcoeff * (1 - edge.stats['P']))
                    else:

                        #Rewriting the new simplificated regret expression with simplecoeff
                        #edge.stats['R'] = edge.stats['R'] - ((value * piadvzi)/pisigprimez) * pisigziz

                        edge.stats['R'] = edge.stats['R'] - (value * simplcoeff * (edge.stats['P']))

    def addNode(self, node):
        self.tree[node.id] = node

