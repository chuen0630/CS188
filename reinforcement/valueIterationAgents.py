# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util


from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            newValues= util.Counter()
            for s in states:
                if self.mdp.isTerminal(s):
                    continue
                actionValues = []
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                    Q = self.computeQValueFromValues(s,a)
                    actionValues.append(Q)
                newValues[s] = max(actionValues)
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          Q(s,a) = Sum_s' T(s,a,s') * (R(s,a,s')+ gamma * V(s') )

        """
        "*** YOUR CODE HERE ***"
        s = state
        a = action
        Q = 0
        s_primes = self.mdp.getTransitionStatesAndProbs(s, a)
        for s_i, T in s_primes:
            R = self.mdp.getReward(s, a, s_i)
            V_s_i = self.values[s_i]
            Q += T * (R + self.discount * V_s_i)
        return Q


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        s = state
        actionValues = util.Counter()
        actions = self.mdp.getPossibleActions(s)
        for a in actions:
            V_sa = 0
            s_primes = self.mdp.getTransitionStatesAndProbs(s, a)
            for s_i, T in s_primes:
                R = self.mdp.getReward(s, a, s_i)
                if self.mdp.isTerminal(s_i):
                    V_sa += R
                    continue
                V_s_i = self.values[s_i]
                V_sa += T * (R + self.discount* V_s_i)
            actionValues[a] = V_sa
        return actionValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            s = states[i % len(states)]
            if self.mdp.isTerminal(s):
                continue
            actionValues = []
            actions = self.mdp.getPossibleActions(s)
            for a in actions:
                Q = self.computeQValueFromValues(s,a)
                actionValues.append(Q)
            self.values[s] = max(actionValues)



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        def getPredeccessors():
            """
            return a dictionary of states:set([predecessors])
            """
            predecessors = dict()
            states = self.mdp.getStates()
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                    successors = self.mdp.getTransitionStatesAndProbs(s, a)
                    for suc, T in successors:
                        if suc in predecessors.keys():
                            predecessors[suc].add(s)
                        else:
                            predecessors[suc] = set([s])
            return predecessors
        def computeMaxQ(state):
            Qs = []
            actions = self.mdp.getPossibleActions(state)
            for a in actions:
                Q = self.computeQValueFromValues(state,a)
                Qs.append(Q)
            return max(Qs)
        "*** YOUR CODE HERE ***"
        predecessors = getPredeccessors()
        priorityQueue = util.PriorityQueue()
        states = self.mdp.getStates()
        for s in states:
            if self.mdp.isTerminal(s):
                continue
            maxQ = computeMaxQ(s)
            diff = abs(self.values[s] - maxQ)
            priorityQueue.update(s,-diff)
        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                return
            s = priorityQueue.pop()
            if self.mdp.isTerminal(s):
                continue
            maxQ = computeMaxQ(s)
            self.values[s] = maxQ
            for p in predecessors[s]:
                maxQ = computeMaxQ(p)
                diff = abs(self.values[p] - maxQ)
                if diff > self.theta:
                    priorityQueue.update(p,-diff)
