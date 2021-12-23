# Reinforcement-Learning


# Maze Reinforcement Learning 

Overview: The goal of the project was to use reinforcement learning to train an agent/mouse to reach the target or achieve a certain task. Reinforcement learning is a system of learning through reward and penalty that helps the agent to follow the right instructions. In maze reinforcement learning, it consists of Markov Decision Process, Exploitation vs Exploration, Bellman Equation,Q-learning, or Q-Sarsa and N- steps look ahead.

# Markov Decision Process has 5 different components:

                  
1. Agent: The trainee that the reinforcement learning is using. In the case of the maze problem, It would be the mouse as the agent. The mouse will be trained through the use of Q-learning.

2. Environment: A place where the agent will be placed for interaction with the maze. 
  or known as 
![image](https://user-images.githubusercontent.com/86531095/147188484-7cc16c39-5ea3-40cc-b725-a00af5924c3e.png)
or
![image](https://user-images.githubusercontent.com/86531095/147188452-360340a3-654e-470d-9aaa-53ef8d336fab.png)

3. State: The agent’s position is known as the state. In this maze problem, it should consist of a set of states for the agent to move around.

4. Action: Are the actions that the agent is able to move around or to do with. For example, the agent is allowed to move right, left, up and down. It doesn’t have to be in order but make sure you use that same order for q_table.

5. Reward : The agent gets rewarded for its action from one state to another for some consequences known as transition reward. 


The final setup of the Environment would be like this:


![image](https://user-images.githubusercontent.com/86531095/147188536-90b91f21-9cfe-42b0-98f7-a5735d0a9389.png)

# Q - Learning:
The agent initial state is (0,0) and the goal of the agent is to reach the cheese state (5,0) through the maze without stepping into any of the blocked blocks. The maze will use Q-learning to train the agent to optimize its action. Q-learning is an algorithm (Off policy) and uses the greedy algorithm to find the optimal action from the current state to the next state. The below is the agent’s at a specific state with its action pair that would give a value known as q value:
					Q(S,A) 
Q-learning has q value function and a q_table. In the case of a 6x6 maze, the q_table would be stored as 6x6x4. A 3 dimension matrix that represents the q value in each state-action pair. The below is the q learning function that helps to format the q_table accordingly. 


First, initialize the q_table to all 0s. It should be something like below:
![image](https://user-images.githubusercontent.com/86531095/147188580-27cce405-e51d-424d-8c8b-43089d3abf98.png)

Each index in the B, C,D, and E columns is known as the q value in a q table.

Before we start the training, we have to declare the agent’s exploration rate and the exploitation rate. In both of the cases for the maze and tic tac toe, The exploration rate is 10% and 90% is exploitation(greedy). In other words, 10% of the time, the agent will take random actions in the maze and 90% of the time, the agent will pick the best action that leads to the goal/target. 
					

# Bellman Equation:
Bellman equation helps us to maximize/optimal policy at a given state with max action of the reward and multiply by the discount factor of the next state: 

			Q(S,A) = R(S,A) + discount_factor * maxQ(S’,Ai)	
 where i is from 0 to all the actions at given state 

The whole q_learning function to update each q value in the q-table is :


Qnew(St,At)  + =  (learning rate) * ( Reward transition +  *maxQ(St+1,A) - Q(St,At))

	The new q value is being updated by the old value + learning rate multiply by the (reward transition from current state to another state by particle action + disfactor   multiply by the max reward of the next state’s best action subtract old value) 

		The whole of q learning/training process should be:
Initialize q-table with all zeros and pick initiate state( starting state of the agent)
Agent picks an action at the current state according to the greedy algorithm.
Store the reward from the current state to the next state with an action taken.
Update the q-value using the q function in the q table.
Repeat steps 2 to 4 until the game is over. 
Create an iteration loop that loop through the number of training sessions from step 2 to 5.

After 5,578 episodes of training, the q_table looks like in 6x6x4 matrix:
![image](https://user-images.githubusercontent.com/86531095/147188616-a6b9facd-9014-4cda-96e0-8adf293e2cc2.png)

Then, the q_table is being used to run the maze again to reach its target accordingly.  From (0,0) to (5,1) would output with its fastest(optimal) path with the best reward output:
![image](https://user-images.githubusercontent.com/86531095/147188633-e93ce8f3-ab16-4834-9613-4c4c940f4a3d.png)

# For Demonstration: 
 
![image](https://user-images.githubusercontent.com/86531095/147188642-db3f98c3-e3da-45ec-8939-8b8b7bf80698.png)


# Subclass : Tic Tac Toe (Group)
As for the Tic Tac Toe problem. The only difference is the environment, reward system and the q-table. Everything remains the same from the maze problem. Tic Tac Toe used Q learning to train the computer/agent to come up with it’s optimal action at the current state.
		
# Environment:
	For the environment, there are 9 states and two players. One that we will use to do the training on and the other to randomly take action across the tic tac toe game board. Our Agent will be the first to take the action start (1,1) on the game board. 
                                             	     

# Reward system: 
	For the reward system, It will only get rewarded towards the end of the game. If the winner is our training agent, it will be rewarded with 1. Lose, will be rewarded with -1 and draw will be rewarded with 0.
# Q_table:
	For the Q-Table, it is going to use a dictionary to store the tuple of the current state and the action for a key value that gives the best optimal action. As it shown below:
![image](https://user-images.githubusercontent.com/86531095/147188693-3ab9e303-ae91-49d7-8255-3cca746e75ee.png)

	
If you're playing randomly against the computer. The computer is able to beat you within 3 steps. As it shown below:
 ![image](https://user-images.githubusercontent.com/86531095/147188721-6bc4bd38-d3b3-4bbc-bad4-3bd1ac34111a.png)

If you're playing your best with the agent. It’s mostly draw games. 
![image](https://user-images.githubusercontent.com/86531095/147188737-06df5997-b83c-4dd1-a5ab-4fef0209479c.png)
![image](https://user-images.githubusercontent.com/86531095/147188749-3e19f6e0-b340-4c78-8447-7db61cd1242d.png)


# Reference sources:
https://towardsdatascience.com/reinforcement-learning-made-simple-part-1-intro-to-basic-concepts-and-terminology-1d2a87aa060
https://www.samyzaf.com/ML/rl/qmaze.html, https://www.analyticsvidhya.com/blog/2020/11/reinforcement-learning-markov-decision-process/ , https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542 
