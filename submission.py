import util, math, random
from collections import defaultdict
from util import ValueIteration



class LunarLanderMDP(util.MDP):
    def __init__(self,args):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        v_not,grid_size,thrust,dtheta_range,time_penalty,fuel_penalty,landing_reward,out_of_bounds_penalty = args 
        self.v_not = v_not
        self.grid_size = grid_size
        self.Landed = False 
        self.thrust = thrust
        self.dtheta_range = dtheta_range
        self.time_penalty = time_penalty
        self.fuel_penalty = fuel_penalty
        self.landing_reward = landing_reward
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.landing_threshold = 3
        self.action_space_size = 0
        self.gravity = 9.8
        self.distancex_penalty = 100
        self.distancey_penalty = 1000
        self.m = 100
        self.landing_count = 0
    
    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        x_not = int(.25*self.grid_size)
        y_not = int(.8*self.grid_size) 
        location = (x_not,y_not)
        velocity = (self.v_not,0)
        theta = 90 
        return (location,velocity,theta)
    

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        actions = [] 
        #location,velocity,theta = state 
        for dtheta in range(-self.dtheta_range,self.dtheta_range+5,5):
            for thrust in [0, self.thrust]:
                actions.append((dtheta,thrust))
        self.action_space_size = len(actions)
        return actions

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        if self.endState(state): return []
        location,velocity,theta = state 

        
        dtheta,thrust = action
        location = self.updateLocation(state,action)
        x,y = location
        velocity = self.updateVelocity(state,action)
        theta += dtheta
        newState = (location,velocity,theta)
        prob = 1.0/self.action_space_size
        reward = self.time_penalty
        reward += self.heuristic(state,newState) 
        if self.terminalState(state): 
            if self.Landed:
                self.Landed = False
                reward += self.landing_reward
            else:
                reward -= self.out_of_bounds_penalty
            return [(None,1,reward)]

        # print(-self.fuel_penalty*(abs(dtheta)+abs(thrust)))
        #return [(newState,1,-self.time_penalty+ -self.fuel_penalty*(abs(dtheta)+abs(thrust)))]
        return [(newState,1,reward)]
        
    def heuristic(self,oldState,newState):
        reward = 0 
        a = 10
        b = 10
        old_location,old_velocity,old_theta = oldState
        new_location,new_velocity,new_theta = oldState
        ydiff = old_location[1] - new_location[1]
        vydiff = old_velocity[1] - new_velocity[1]
        #penalize x velocity for straying from zero 
        reward -= .10*abs(new_velocity[0])

        reward -= .5*abs(new_location[0]-(self.grid_size/2))

        reward -= .5*abs(new_location[1])

        """
        if ydiff > 0: #heading downwards
            reward += a*ydiff
            if vydiff > 0:
                reward += b*ydiff
        """
        return reward
        
    def norm(self,a,b):
        pass 
        
    def endState(self,state):
        if state == None: return True
        return False 
    
    def updateLocation(self,state,action):
        location,velocity,theta = state 
        dtheta, thrust = action
        x,y = location
        vx,vy = velocity
        xnew = x + vx + .5*(math.cos(math.radians(theta))*thrust)/self.m
        ynew = y + vy + .5*(-self.gravity*self.m + math.sin(math.radians(theta))*thrust)/self.m
        return (int(xnew),int(ynew))
    
    def updateVelocity(self,state,action):
        location,velocity,theta = state 
        dtheta,thrust = action
        vx,vy = velocity
        vxnew = vx + math.cos(math.radians(theta))*thrust/self.m
        vynew = vy + (-self.gravity*self.m + math.sin(math.radians(theta))*thrust)/self.m
        return (int(vxnew),int(vynew))
    
    def hasLanded(self,newState):
        x,y = newState[0]
        vx,vy = newState[1]
        if y <= 10 and y > 0:
            if abs(vx) < self.landing_threshold and abs(vy) < self.landing_threshold:
                self.Landed = True 
                self.landing_count +=1
                return True 
        return False 
    
    def discount(self):
        return 1

    def terminalState(self,state):
        if self.hasLanded(state): return True 
        location,velocity,theta = state
        if location[0] >= self.grid_size or location[1] >= self.grid_size: return True 
        if location[0] <= 0: return True 
        if location[1] <= 0 : return True 
        return False

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    cardValues = [3,21]
    multiplicity = 10
    mdp = LunarLanderMDP(env)
    return mdp 
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    def performGradientStep(self,eta,prediction,target,state,action):
        for f,v in self.featureExtractor(state,action):
            self.weights[f] = self.weights[f] - (eta*(prediction - (target)))*v



    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        # if newState == None: continue  #TODO CONSIDER THIS?
        eta = self.getStepSize()
        prediction = self.getQ(state,action)
        actions_prime = self.actions(newState)
        Q_nexts = []
        if newState == None: actions_prime = []
        
        for a_prime in actions_prime:
            Q_next_current = self.getQ(newState,a_prime)
            Q_nexts.append(Q_next_current)
       
        max_Q_next = max(Q_nexts) if newState != None else 0
        target = reward + self.discount*(max_Q_next)
        self.performGradientStep(eta,prediction,target,state,action)
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
#smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
#largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE
   
    ql = QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       featureExtractor)
    q_rewards = util.simulate(mdp, ql, 30000)


    avg_reward_q = float(sum(q_rewards))/len(q_rewards)

    vi = ValueIteration()
    vi.solve(mdp)

    rl = util.FixedRLAlgorithm(vi.pi)
    vi_rewards = util.simulate(mdp,rl,30000)

    avg_reward_vi = float(sum(vi_rewards))/len(vi_rewards)

    ql.explorationProb = 0 
    ql_pi = {}    
    for state,_ in vi.pi.items(): 
        ql_pi[state] = ql.getAction(state)
    p_vi = vi.pi

    diff = 0
    for state in vi.pi.keys():
        if vi.pi[state] != ql_pi[state]: diff +=1 

    print("difference",diff,"over " + str(len(p_vi.keys())) + " states")
    print("percentage diff ",float(diff)/len(p_vi.keys()))
    print("avg_reward_diff",avg_reward_q - avg_reward_vi)






    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
    result = []
    featureKey = (state,action)

    s = (action,total)
    result.append((s,1))

    if counts != None:
        l_counts = list(counts)
        cardsExist = list(counts)
        for i in range(len(counts)):
            if counts[i] > 0: cardsExist[i] = 1 
            s = (action,i,counts[i])
            result.append((s,1))

        s = (action,tuple(cardsExist))
        result.append((s,1))


    return result 
    # END_YOUR_CODE
