# Simple hidden markov model calculation
# Created by Craig Duncan 22 December 2018
# TO DO: put this in a jupyter notebook
# This is a worked example of hidden markov model
#
#  One understood, the calculation process is not that difficult, but involves a reasonable amount
#  of book-keeping beyond the simplest examples
#
# Even if the actual state of a markov chain at any time step is unknown
# the initial probability of observing a symbol in each state is dependent on state not time
# (these priors/beliefs for observations need not be varied, though learning algos might do so)
#
# Stage 1, calculating 'alphas':
# we are simply calculating time-updated probabilities of observations 
# we are storing these at each state, using probability of arriving there from any previous state
# this reduces the intractability of the combinations of possible states and observations
#
# --- Initial setup of markov chain probabilities
#initial value probabilities.  Data structure required : array or matrix
#
# The main advantage of numpy matrices is that they provide a convenient notation for matrix multiplication.
# in the HMM alpha calcs do we need, or is it convenient to do matrix multiplication?
#
# python only has lists built in, but you can use them with numbers.
# if you include lists within a list, you end up with something like a matrix
# e.g. lst = [[1,2,3],[4,5,6]]
# numpy library creates a multidimensional object described as a matrix.  Not used here
##
 
# initial probabilities of being in each state.
initprob = [0.25,0.25,0.25,0.25]  #effectively a 1 x 4 list = vector row
# test what type python has given to this:
#init prob is type list, initprob[0] is type float
#
# Markov chain transition probabilities
# this is the markov chain for states 1 to 4 represented as a matrix (4 x 4) with 
# transition probabilities from the row headings (states FROM) to the col heads (states TO)
# FROM states run across as rows (each list is a row).  TO states are columns in these lists
transprob = [[0.1,0.1,0.8,0],[0.8,0.1,0,0.1],[0.1,0,0.1,0.8],[0,0.8,0.1,0.1]]  #4 row x 4 col

# Observation probabilities for states (known, even if states not observable) 
#
# Matrix style list to hold the probability of 'win' or 'loss' observation outcomes
# given states 1 through 4
# you could visualise this as a vertical matrix instead if that is easier
symbols = [[0.2,0.8],[0.5,0.5],[0.1,0.5],[0.8,0.2]] # 4 rows x 2 columns in list form
##
# matrix to hold alpha values for each state (cols) for each time step (rows)
# this could be dealt with differently, but here it is an array of state lists.  
# In formulas below at each time step I pull out state list to record alphas for a given obs
alpha=[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],] # prepare a list that will contain the four alphas for the observations
#
# A list of actual observations at each time step.  
# Coded to correspond to index of columns in symbols matrix.
# We are using a given sequence of observations, we are not 'running' a simulation (like Monte Carlo)
obs=[0,0,1,1] # index is time step.  0=wins, 1= losses. See Symbols matrix.
fwdprob = [0.0,0.0,0.0,0.0] # list to hold sum FORWARD probs to each state

# define a function to calculate the forward state transition probabilities
# (unless in a class, need to define the function before you use it)
# also, unless defined in a class, no need to put 'self' as first argument of function
# in a class, the 'self' argument effectively passes in the class as an argument?

def getfwdtrans(currentstate):
	fwdprob=0.0
	for prevstate in range (0,4): #cycle from states index 0 to 3
		fwdprob = fwdprob + transprob[currentstate][prevstate]
	# print ("State {} has forward prob:{}".format(currentstate,fwdprob))
	return fwdprob

# the alphas to be calculated are a set of probability values for each state at each time step
# loop to recursively calculate alpha values at each time step.
#
for time in range(0,4):
	symbolprob = obs[time]  # type int
	if time == 0:
		lastalpha = initprob
	else:
		lastalpha = alpha[time-1]
	
	for thisstate in range(0,4): # four states here
		ftp = getfwdtrans(thisstate)
		# this is the main recursive formula for alphas:
		fwdprob[thisstate] = lastalpha[thisstate]*ftp*symbols[thisstate][symbolprob] #state 1, obs 1
	#put a list of alphas for each state, this time step
	alpha[time]=[fwdprob[0],fwdprob[1],fwdprob[2],fwdprob[3]]  
print(alpha)


# --- end of alpha calculations (FORWARD) up to required time steps ---
#
# HMM Problem 1: what is the most likely sequence of states? Dyn Programming or HMM sequence?
# Using a dynamic programming approach find optimum state probability at each time step

# Search for the highest probability (alpha) state in each time step
# This is really what is normally called the 'argmax' function
# two steps: highest value alpha at each time step; then repeat over all time
def printoptimumstates():
	optimumalpha = [0.0,0.0,0.0,0.0] #store the states with highest probability as ints
	optimumstate=[0,0,0,0]
	for time in range(0,4):
		for state in range(0,4):
			if alpha[time][state]>optimumalpha[time]:
				optimumalpha[time]=alpha[time][state]
				optimumstate[time]=state+1 #store the first state as 1, not the index value
		print ("Optimum alpha state at time {} is state:{} value:{}".format(time,optimumstate[time],optimumalpha[time]))


# call the function
printoptimumstates()

# Problem 2
# an actual solution with valid states must check that the actual [inferred] state transitions can work 
# for example, State 2 cannot be reached directly from State 3 so alphas associated with these consecutive states are problematic.
# Some people try and find those paths with the max number of the argmax states in a valid path 
# but the most common is to find the single best set of actual states given the observations.
# (this is known as the Viterbi algorithm)
# This is actually a different procedure to the one that solved for problem 1, though there are 
# similarities.
# 
