"""
Name: Huihong Zheng
Resources Links: https://www.samyzaf.com/ML/rl/qmaze.html ,
https://www.youtube.com/watch?v=77m1j2qYPRg&t=1062s
https://www.baeldung.com/cs/epsilon-greedy-q-learning,
https://www.analyticsvidhya.com/blog/2021/04/q-learning-algorithm-with-step-by-step-implementation-using-python,
 https://towardsdatascience.com/reinforcement-learning-made-simple-part-1-intro-to-basic-concepts-and-terminology-1d2a87aa060
"""

import numpy as np
import random

#x is a blocked object and empty strings are the index that the agent is able to move to
#6x6 maze
maze_size = (
    ["","","","","x",""],
    ["","","x","","",""],
    ["","","x","x","x",""],
    ["x","","x","","",""],
    ["","x","","","x",""],
    ["","","","","","x"])

keep_track_counter = (
    [0,0,0,0,0,0,],
    [0,0,0,0,0,0,],
    [0,0,0,0,0,0,],
    [0,0,0,0,0,0,],
    [0,0,0,0,0,0,],
    [0,0,0,0,0,0,],
    [0,0,0,0,0,0,]
)

#Declare class name EnvironmentMaze
#agent will be at starting position of (0,0)
#goal: is the target that we want the agent to achieve
class EnvironmentMaze:
    def __init__(self):
        self.agent = (0,0) #Initalize the starting postion of the agent at (0,0) if there is reset agent position
        self.maze = np.array(maze_size)  #Set up the maze
        self.goal =  (5,0) #The target 
        self.blocks = self.free_blocks() #Setting up the movement within the free cells
        self.visted = [] #Empty visted cells
        self.current_reward = 0 
        self.game_over = "no"
        self.whole_maze = self.all_blocks()   #Will not be using it / probably be removed
        #To keep in track of how many revisted cells
        self.countvisted = np.array(keep_track_counter) 

    #Resetting everything to original except the current position of the agent
    def reset(self,agent_start):
        self.agent = agent_start
        self.visted = []
        self.current_reward = 0
        self.game_over = "no"

    #Creating the free blocks that allows the agent to move around with
    def free_blocks(self):
        moveable_block =[]
        for row in range(0,len(self.maze)):
            for column in range(0,len(self.maze[0])):
                if self.maze[row,column] !="x":
                    moveable_block.append((row,column))
        return moveable_block
    
    #All the blocks inside the maze with
    #@full_maze
    def all_blocks(self):
        full_maze = []
        for row in range(0,len(self.maze)):
            for column in range (0,len(self.maze[0])):
                full_maze.append((row,column))
        return full_maze

    """
    Take an action in a current state with feedbacks
    Action that are being repeated at certain state or been visted will be a harder punishment 
    Check to see if the currentstate of the agent is in the free blocks, blocked cells is game_over(lose),  outside maze will be punished same with blocked cells,
    and everything else is game over
    """
    def action(self,direction,currentstate):
        self.direction =direction
        row, column = currentstate
        maze_row,maze_column = self.maze.shape
        if currentstate in self.blocks and self.direction == "right":
            column = column +1
            if (row,column) == self.goal:
                self.agent = (row,column)
                self.current_reward = 200
            elif (column >=0 and column <= maze_column-1 and 
            [row,column] not in self.visted and (row,column) in self.blocks):
                self.agent = (row,column)
                self.visted.append([row,column])
                self.current_reward = -1
            elif (column >=0 and column <= maze_column-1 and 
            [row,column] in self.visted and (row,column) in self.blocks) :
                self.agent = (row,column)
                self.countvisted[row,column] +=1
                self.current_reward = -1
            elif (column >=0 and column <= maze_column-1):
                self.current_reward = -100
                self.game_over = "yes"
                self.agent = (row,column)
            #outside of the maze
            else:
                self.current_reward = -100
        elif currentstate in self.blocks and self.direction == "left":
            column = column -1
            if (row,column) == self.goal:
                self.agent = (row,column)
                self.current_reward = 200
            elif (column >=0 and column <= maze_column-1 and 
            [row,column] not in self.visted and (row,column) in self.blocks):
                self.agent = (row,column)
                self.visted.append([row,column])
                self.current_reward = -1
            elif (column >=0 and column <= maze_column-1 and 
            [row,column] in self.visted and (row,column) in self.blocks):
                self.agent = (row,column)
                self.countvisted[row,column] +=1
                self.current_reward = -1
            #inside the maze but inside blocked cell
            elif (column >=0 and column <= maze_column-1):
                self.current_reward = -100
                self.game_over = "yes"
                self.agent = (row,column)
            else:
                self.current_reward = -100
        elif currentstate in self.blocks and self.direction == "up":
            row = row -1
            if (row,column) == self.goal:
                self.agent = (row,column)
                self.current_reward = 200
            elif (row >=0 and row <=maze_row-1 and 
            [row,column] not in self.visted and (row,column) in self.blocks):
                self.agent = (row,column)
                self.visted.append([row,column])
                self.current_reward = -1
            elif (row >=0 and row <=maze_row-1 and 
            [row,column] in self.visted and (row,column) in self.blocks):
                self.agent = (row,column)
                self.current_reward = -1
            elif (row >=0 and row <= maze_row-1):
                self.current_reward = -100
                self.game_over = "yes"
                self.agent = (row,column)
            #outside of the maze
            else:
                self.current_reward = -100
        elif currentstate in self.blocks and self.direction == "down":
            row = row + 1
            #reached goal 
            if (row,column) == self.goal:
                self.agent = (row,column)
                self.current_reward = 200
            #first visited block
            elif (row >=0 and  row<= maze_row-1 and 
            [row,column] not in self.visted and (row,column) in self.blocks):
                self.agent = (row,column)
                self.visted.append([row,column])
                self.current_reward = -1
            #revisited blocked 
            elif (row >=0 and row <= maze_row-1 and 
            [row,column] in self.visted and (row,column) in self.blocks):
                self.agent = (row,column)
                self.countvisted[row,column] +=1
                self.current_reward = -1            
            #Within the maze but stepped on the blocked cells
            elif (row >=0 and row <= maze_row-1):
                #Reward where it's -100, ran into a blocked cell
                self.agent = (row,column)
                self.current_reward = -100
                self.game_over = "yes"
            #After taking the action is outside of the maze
            else:
                self.current_reward = -100
        #everything else will be game_over 
        else:
            self.game_over = "yes"

    """"Will ignore this function for now"""
    #To prevent infinite looping in the same cell being revisted 20x
    def count_the_visted(self):
        for row in range(0,len(self.countvisted)):
            for column in range(0,len(self.count_visted[0])):
                if self.count_visted[row,column] >=20:
                    self.game_over ="yes"

    """
    To check if the current agent have achieved the goal or 
    ran into a blocked cell/block inside the maze
    """
    def game_case(self):
        if self.agent == self.goal:
            return "win"
        if self.game_over == "yes":
            return "lose"
 
    """
    actions 0-right , 1-left , 2-up , and 3-down
    A function that will return the actions that the agent allow to make
    at current state
    """
    def moveable_action(self):
        row,column = self.agent
        valid_action = []
        if (row,column+1) in self.blocks:
            valid_action.append(0)
        if (row ,column-1) in self.blocks:
            valid_action.append(1)
        if (row +1,column) in self.blocks:
            valid_action.append(2)
        if (row -1,column) in self.blocks:
            valid_action.append(3)
        return valid_action

"""Q-Table: 3D table with 6x6x4
First two dimension are the current state of the agent
The 3rd dimension is the actions
"""
def create_q_table():
    q_table = []
    for a in range(6):
        q_table.append([])
        for b in range(6):
            q_table[a].append([])
            for c in range(4):
                q_table[a][b].append(0)
    return q_table

"""
#Exploration will be 10% and exploiation will be 90% of the time  
A function that will take in the object and the q table
10% of the time it will randomly pick an action and 90% will pick the greedy action(max reward)
@set_of_actions: are actions that the agent can move given in it is current state.
np.argmax will return max reward action
@return action = is the right,left, up or down depending if statement
"""
def eplision_greedy(Maze,q_table,exploration):
    set_of_actions = Maze.moveable_action()
    curr_row,curr_column = Maze.agent
    if np.random.rand() > exploration:
        #Need to find the best action in a Q value in that state
        action= np.argmax(q_table[curr_row][curr_column])
    else:
        action = random.choice(set_of_actions)
    return action 

"""
It will take two parameter the maze and action number
Will be using the agent to move around the maze with specific action(direction)
be using object methods to perform the action
"""
def take_action(Maze,action):
    if action == 0:
        direction = "right"
        currentstate = Maze.agent
        Maze.action(direction,currentstate)
    if action == 1:
        direction = "left"
        currentstate = Maze.agent
        Maze.action(direction,currentstate)
    if action == 2:
        direction = "up"
        currentstate = Maze.agent
        Maze.action(direction,currentstate)
    if action ==3:
        direction = "down"
        currentstate = Maze.agent
        Maze.action(direction,currentstate)

"""
q algorithm that trains the agent into best actions with constantly updating q table
Takes in 3 parameter(q_table, the object class , and number of trainning epsideos)
discount_factor and aplha(learning rate ) will be set to 1
#https://medium.com/analytics-vidhya/q-learning-expected-sarsa-and-comparison-of-td-learning-algorithms-e4612064de97
#That link will be the q leanring off policy algorithm 
"""
def q_train(q_table,Maze,number_ep):
    discount_factor = 1 
    aplha = 1 #Learning rate 
    memory_track = []       #keep in track of wins and loses 
    exploration = .10    #10% will be exploration and 90% will be exploitation
    #loop through the number of epsideos
    for eps in range(number_ep):
        game_over = False
        #Taking random choice within the freeblock of the environment(the blocks that the agent able to move around)
        #Reset the agent into that position
        agent_position = random.choice(Maze.blocks)
        Maze.reset(agent_position)
        while game_over != True:
            #constantly check the game status to see if the agent have win or lose for each action of the loop
            game_status = Maze.game_case()
            if game_status == "win":
                memory_track.append("win")
                game_over = True
            elif game_status == "lose":
                memory_track.append("lose")
                game_over = True
            else:
                game_over = False
            #The agent will have 10% exploration while 90% will be exploiation 
            curr_row , curr_column = Maze.agent
            action = eplision_greedy(Maze,q_table,exploration)
            take_action(Maze,action)
            #transition reward from current state to next state from taken action
            transition_reward = Maze.current_reward #Reward from one state to next state
            new_row,new_column = Maze.agent #The very next state after the action taken 
            current_q_value = q_table[curr_row][curr_column][action]   #Store the previous q value with the action taken
            #update the new q_value = current q + learning rate*( r +(gamma +maxq(nextstate) -current q))
            q_table[curr_row][curr_column][action] = current_q_value +  (aplha* (transition_reward+ 
            (discount_factor*(np.max(q_table[new_row][new_column]))) - current_q_value))
        print(f"******************* Right now is on episode: {eps}")
        Winning_streak = 80
        mem_size = len(memory_track)
        #When the size of the memory is equal to 80:
        #We will check if the whole sequences of that memorytrack is all wins
        #Compute the percentage of total wins over Winning streak
        #if the percentage is equal to 1 then prints out it's converges on that number of eps with optimal solution
        if mem_size == Winning_streak :
            total_win = 0
            converge = False
            for x in range(0,mem_size):
                if memory_track[x] == "win":
                    total_win +=1
            win_percent = total_win / Winning_streak
            memory_track.pop(0) #Delete the very first memory_track to keep in track of newer ones
            if win_percent ==1:
                converge = True
            if converge:
                print(f"\n\n ****************************Number of episodes it took to win: {eps}******************************\n")
                print("\n this is Q-learning : one step off policy \n ")
                print(memory_track)
                break
            else:
                pass
"""
https://www.geeksforgeeks.org/sarsa-reinforcement-learning/
following one step SARSA formula 
stay with .1 Learning rate
"""
def q_train_sarsa(q_table,Maze,number_ep):
    discount_factor = .9
    aplha = .1
    exploration = .10      #10% will be exploration and 90% will be exploitation
    memory_track = []
    #Loop through number of eps
    for eps in range(number_ep):
        game_over = False
        agent_position = random.choice(Maze.blocks) 
        Maze.reset(agent_position)
        action = eplision_greedy(Maze,q_table,exploration) #get the very first action
        curr_row , curr_column = Maze.agent #get the current state of the agent 
        while game_over != True:
            take_action(Maze,action) #Perform the action 
            transition_reward = Maze.current_reward #The transition reward from current state to very next state from action taken
            new_row,new_column = Maze.agent #Get the very next state 
            action_2 = eplision_greedy(Maze,q_table,exploration) #Take the very next state max action but not to perform the action
            # print(f"Action = {action} Agent postion after action: {Maze.agent} Transition reward: {transition_reward}")
            currt_value = q_table[curr_row][curr_column][action] #store the q-value of the previous state with action taken to very next state
            # print(current_q_value)

            #It didn't perform the action for the very next state. It took the very next action to store it
            #update the new q_value = current q + learning rate*( r +(gamma +q(nextstate,with nextstate action) -current q))
            q_table[curr_row][curr_column][action] = currt_value + (aplha* 
            (transition_reward+(discount_factor*(q_table[new_row][new_column][action_2])) - currt_value))
            game_status = Maze.game_case()
            if game_status == "win":
                memory_track.append("win")
                game_over = True
            elif game_status == "lose":
                memory_track.append("lose")
                game_over = True
            else:
                game_over = False
                #set the next action to be action2 that haven't performed at that very next state
                action = action_2
                #set the current state of the agent to the very next state
                curr_row, curr_column = new_row, new_column
        print(f"******************* Right now is on episode: {eps}")
        Winning_streak = 80
        mem_size = len(memory_track)
        #When the size of the memory is equal to 80:
        #We will check if the whole sequences of that memorytrack is all wins
        #Compute the percentage of total wins over Winning streak
        #if the percentage is equal to 1 then prints out it's converges on that number of eps with optimal solution
        if mem_size == Winning_streak :
            total_win = 0
            converge= False
            for x in range(0,mem_size):
                if memory_track[x] == "win":
                    total_win +=1
            win_percent = total_win / Winning_streak
            memory_track.pop(0) #Delete the very first memory_track to keep in track of newer ones
            if win_percent ==1:
                converge = True
            if converge ==1:
                print(f"\n\n ****************************Number of episodes it took to win: {eps}******************************\n")
                print("\n this is one step SARSA on policy \n")
                print(memory_track)
                break
            else:
                pass



"""
******THis in Process fixing*********
https://lcalem.github.io/blog/2018/11/19/sutton-chap07-nstep#73-n-step-off-policy-learning-by-importance-sampling
Following the 7.2 n step Sarsa
N step Saras algorithm
"""
def q_train_n(q_table,Maze,number_ep):
    discount_factor =.9
    aplha = .2
    exploration = .10 #10% will be exploration and 90% will be exploitation
    memory_track = [] #keep in track of the "win" or "lose"
    for eps in range(number_ep):
        n = 7    #Number of storage
        previous_states = [0] * n
        previous_actions = [0] * n
        previous_rewards = [0] * n
        game_over = False
        number_of_step = 0 #Starting step 
        T = float('inf')
        agent_position = random.choice(Maze.blocks)  
        Maze.reset(agent_position)
        action = eplision_greedy(Maze,q_table,exploration) #Choose an action 
        previous_states[0] = Maze.agent #Store the initialze position of the agent
        previous_actions[0] = action #Store the initailaze action 
        # print("First position: ",Maze.agent)
        # print("First Action: ",action)
        while game_over != True:
            take_action(Maze,action) #Perform the action 
            previous_rewards[(number_of_step+1)%n] =Maze.current_reward #Store the reward of Reward(step+1)
            previous_states[(number_of_step+1)%n] = Maze.agent #Store the State(step+1)
            # print(Maze.current_reward)
            # print("Action2: ",action)
            # print(Maze.agent)
            # print(f"Action = {action} Agent postion after action: {Maze.agent} Transition reward: {Maze.current_reward}")
            game_status = Maze.game_case()
            #If state(step+1) is terminal then T <-step +1
            if game_status == "win":
                game_over = True
                memory_track.append("win")
                T = number_of_step+1
            elif game_status == "lose":
                game_over = True
                memory_track.append("lose")
                T = number_of_step+1
            #Else select and store an action(step+1)
            else:
                game_over = False
                action = eplision_greedy(Maze,q_table,exploration)
                previous_actions[(number_of_step+1)%n] = action   #Store the action of (step +1)
            π = number_of_step - n +1  #When the update happens
            if π >= 0:
                stop_at = min(π+n,T)
                expected_return = 0
                #Adding up all the rewards from π+1 until mini of (π+n,T+1)
                # R(t+1) + discount factor(R(t+2)).........
                for i in range(π+1,stop_at+1):
                    expected_return =  expected_return+ (discount_factor**(i-π-1)) *previous_rewards[i%n] 
                #if T is greater than π +n then, discount^n*state[π+n] with action [π+n]
                if π +n <T:
                    state_row, state_column = previous_states[(π+n)%n]
                    action_taken = previous_actions[(π+n)%n]
                    expected_return = expected_return+ (discount_factor**n*q_table[state_row][state_column][action_taken])
                #update the q value from current state π with that state action taken 
                #using the rewards from n steps 
                state_row, state_column = previous_states[π%n]
                action_taken = previous_actions[π%n]
                current_value =q_table[state_row][state_column][action_taken]
                q_table[state_row][state_column][action_taken] = current_value + aplha*(expected_return-current_value)
            number_of_step +=1        
        
        #Update the remaining of the steps 
        #number of step will get extra one from while loop because it will update the remainning steps
        #until  π = T -1
        for π in range(number_of_step-n+1,T):
            if π >=0: #Only wants to update the steps between 0 to N steps remaining
                stop_at = min(π+n,T)
                expected_return = 0
                #Adding up all the rewards from π+1 until mini of (π+n,T+1)
                # R(t+1) + discount factor(R(t+2)).........
                for i in range(π+1,stop_at+1):
                    expected_return = expected_return+ (discount_factor**(i-π-1)) *previous_rewards[i%n] 
                if π +n <T:
                    state_row,state_column = previous_states[(π+n)%n]
                    action_taken = previous_actions[(π+n)%n]
                    expected_return = expected_return + (discount_factor**n*q_table[state_row][state_column][action_taken])
                #update the q value from current state π with that state action taken 
                #using the rewards from n steps 
                state_row, state_column = previous_states[π%n]
                action_taken = previous_actions[π%n]
                current_value = q_table[state_row][state_column][action_taken]
                q_table[state_row][state_column][action_taken] = current_value+ aplha*(expected_return-current_value)
        print(f"******************* Right now is on episode: {eps} ||  Number of steps: {number_of_step}")
        Winning_streak = 20
        mem_size = len(memory_track)
        #When the size of the memory is equal to 80:
        #We will check if the whole sequences of that memorytrack is all wins
        #Compute the percentage of total wins over Winning streak
        #if the percentage is equal to 1 then prints out it's converges on that number of eps with optimal solution
        if mem_size == Winning_streak :
            total_win = 0
            converge = False
            for x in range(0,mem_size):
                if memory_track[x] == "win":
                    total_win +=1
            win_percent = total_win / Winning_streak
            memory_track.pop(0) #Delete the very first memory_track to keep in track of newer ones
            if win_percent ==1:
                converge = True
            if converge:
                print(f"\n\n ****************************Number of episodes it took to win: {eps}******************************\n")
                print("\n This is N step look ahead \n ")
                print(memory_track)
                break
            else:
                pass                       
"""
function that convert the actions into words format
returns the direction of given action
"""
def direction_with_word(action):
    if action == 0:
        return "right"
    if action == 1:
        return "left"
    if action == 2:
        return "up"
    if action == 3:
        return "down"

"""
a function that will test how effective our trainning is to get the optmial path
@path will store the state of the agent (current state of the agent such as (4,3) and so on.....)
@direction will store the moving direction of the agent( such as right,left,up,down)
return both the path and direction
"""
def run_test(q_table,Maze,position):
    exploration = 0
    path = []
    direction = []
    Maze.reset(position)
    path.append(position)
    game_over = False
    count = 0
    while game_over != True:
        game_status = Maze.game_case()
        if game_status == "win":
            game_over = True
        elif game_status == "lose":
            game_over = True
        else:
            game_over = False
            action = eplision_greedy(Maze,q_table,exploration)
            take_action(Maze,action)
            path.append(Maze.agent)
            direction_sign = direction_with_word(action)
            direction.append(direction_sign)
        count += 1
        #incase infinite looping break the looping
        if count > 100:
            print("\n *************Not working properly********************")
            break
    return path,direction


"""
Print out a 3D Q-table 6x6x4 
[row][col][action]
It will be from row 0 to row 5
"""
def print_nice_table(q_table):
    for x in range(0,len(q_table)):
        print(q_table[x])

"""
The main driver of the program
create q_table and setting up the evironment, agent, trainning, and action
path will print out the start state of the agent to the goal
direction will print out the direction of the agent is moving within the maze to achieve the goal
"""
if __name__ == '__main__':
    q_table = create_q_table()
    Maze = EnvironmentMaze()
    """One Step Q Learning Works off policy"""
    q_train(q_table,Maze,250000)

    print("\n\n")
    """One Step SARSA Works on policy stay with -100 rewards for blocks and outside the maze"""
    # q_train_sarsa(q_table,Maze,250000)



    """*******THis in Process fixing*********
    N step look ahead / doesn't always work """ 
    """ not working correctly for different positions / Run it a few times on the same position. there would be differences. It work sometimes and it doesn't sometimes"""
    # q_train_n(q_table,Maze,35000)

    print('\n\n')
    print_nice_table(q_table)
    print('\n\n')
    start_position = (0,0)
    path,direction = run_test(q_table,Maze,start_position)
    print(path)
    print("\n")
    print(direction)
