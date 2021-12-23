"""
Name: Huihong Zheng
Resources Links:https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542,
https://nestedsoftware.com/2019/07/25/tic-tac-toe-with-tabular-q-learning-1kdn.139811.html,
https://medium.com/@carsten.friedrich/part-3-tabular-q-learning-a-tic-tac-toe-player-that-gets-better-and-better-fa4da4b0892a,
https://www.govindgnair.com/post/solving-tic-tac-toe-with-reinforcement-learning/
"""
import pickle #helps to save policy and load policy to/from a file
import numpy as np
import random

#Setting up an empty 3x3 matrix for tic tac toe
Board = (
    ["","",""],
    ["","",""],
    ["","",""])

class TicTacoToe:
    def __init__(self):
        self.player1 = "❌"
        self.player2 = "⭕"
        self.board = np.array(Board)
        self.game_status = False
        self.winner = ""
        self.reward = 0
        self.loser_reward = 0 

    #reset game board
    def reset_board(self):
        self.player1 = "❌"
        self.player2 = "⭕"
        self.board = np.array(Board)
        self.game_status = False
        self.winner = ""
        self.reward = 0
        self.loser_reward = 0  
        self.board[1][1] = self.player1    #starting position for training agent
    
    #print out the game board
    def print_board(self):
        for x in range(0,len(self.board)):
            print(self.board[x])      

    #update the game board after an action was performed and check if any winners
    def update_board(self,location,player):
        if location in self.free_space():
            self.board[location] = player
            self.check_winner(player)
        else:
            print("There is no such space")

    """
    Check the available space in the game board
    @free_box: return a list array of the available space/spaces
    """
    def free_space(self):
        free_box = []
        for row in range(0,len(self.board)):
            for column in range(0,len(self.board[0])):
                if self.board[row][column] != "❌" and self.board[row][column] !="⭕":
                    free_box.append((row,column))
        return free_box

    #Check whether if the loser is our agent. If it is our trainee agent then reward -1 for losing
    def opponent_lost(self,Winner_player):
        if Winner_player != self.player1:
            self.loser_reward =-1
    
    def check_winner(self,player):
        #Check Columns
        if self.board[0][0] == player and self.board[1][0] == player and self.board[2][0] == player:
            self.game_status = True
            self.winner = player
            self.reward = 1
            self.opponent_lost(player)
        elif self.board[0][1]==player and self.board[1][1]==player and self.board[2][1]== player:
            self.game_status = True
            self.winner = player
            self.reward =1
            self.opponent_lost(player)
        elif self.board[0][2] == player and self.board[1][2]== player and self.board[2][2]== player:
            self.game_status = True
            self.winner = player
            self.reward =1
            self.opponent_lost(player)
        #Check rows
        elif self.board[0][0]== player and self.board[0][1]== player and self.board[0][2]== player:
            self.game_status = True
            self.winner = player
            self.reward = 1
            self.opponent_lost(player)
        elif self.board[1][0]== player and self.board[1][1]== player and self.board[1][2]== player:        
            self.game_status = True
            self.winner = player
            self.reward = 1
            self.opponent_lost(player)
        elif self.board[2][0]== player and self.board[2][1]== player and self.board[2][2]== player:
            self.game_status = True
            self.winner = player
            self.reward = 1
            self.opponent_lost(player)
        #check diagonal 
        elif self.board[0][0]== player and self.board[1][1]== player and self.board[2][2]== player:
            self.game_status = True
            self.winner = player
            self.reward =1 
            self.opponent_lost(player)
        elif self.board[0][2]== player and self.board[1][1]== player and self.board[2][0]== player:
            self.game_status = True
            self.winner = player
            self.reward = 1
            self.opponent_lost(player)
        #else if the they are draw
        elif not self.free_space():
            self.game_status = True
            self.winner = "Draw"
            self.reward = 0
        #there still space in the tic tac toe board and game is not over yet
        else:
            self.game_status = False

    #Return whether the game is over or not
    def check_game_status(self):
        if self.game_status:
            return True
        else:
            return False


"""
Randomly picks an action that is free in the tic tac toe state
@return the picked action
"""
def random_action_p2(game_board,player):
    free_positions = game_board.free_space()
    action = random.choice(free_positions)
    return action

"""
Helps to initalize the q_table with current state-action pair to 0 if they didn't exist 
and going to be used in the next line.
@return the updated q (q_table) dictionary
"""
def check_Q(q, state, action):
    if state not in q:
        q[state] = {}
        q[state][action] = 0
    elif state in q:  
        if action not in q[state]:
            q[state][action] = 0
    else:
        pass

    return q


"""
A greedy algorithm that helps the agent to find the best action of the current board
Exploration(random) is 10% and 90% expoliation
@action: returns the best position for the current state of the board
"""
def greedy_action(game_board,q):
    free_positions = game_board.free_space()
    exploration = .10
    currentState = tuple(game_board.board.reshape(1,-1)[0])
    if np.random.rand() > exploration:
        #Check all actions to find the best action leads to the best reward
        max_value = -np.inf
        for position in free_positions:
            free_action = tuple(position)
            q = check_Q(q,currentState,free_action)
            temp_value = q[currentState][free_action]
            if temp_value > max_value:
                max_value = temp_value
                action = position
    else:
        action = random.choice(free_positions)
    return action



"""
Q learning will be used to train the agent againist an random action player.
learning rate: 1
Discount factor = .9
reward will be rewarded towards the end of the game. If the trainnee is a winner +1. If loses -1 and draw is 0.
@memory_track: array hold the previous history of winning/losing/draw scores.
"""
def Q_learning(game_board, number_ep):
    discount_factor = .9
    aplha = 1
    memory_track = []
    q = {} #q table dictionary   
    for eps in range(number_ep):
        game_over= False
        game_board.reset_board()
        current_state = tuple(game_board.board.reshape(1,-1)[0])
        action_player1 = tuple((1,1))
        while game_over != True:
            #Player 2 take action and update the board
            game_player2= game_board.player2
            action_player2 = random_action_p2(game_board,game_player2)
            game_board.update_board(action_player2,game_player2)
            new_state = tuple(game_board.board.reshape(1,-1)[0])
            #Check game status
            game_status = game_board.check_game_status()
            if game_status: 
                if game_board.winner == "❌":
                    transition_reward = game_board.reward 
                    action1 = tuple(action_player1)
                    q = check_Q(q,current_state,action1) #Add to dictionary only if state and action pair didn't exist
                    current_value = q[current_state][action1]
                    free_positions = game_board.free_space()
                    max_new_state = 0 
                    #Find the max of the next state for maxQ(St+1,A)
                    for position in free_positions:
                        free_action = tuple(position)
                        q = check_Q(q,new_state,free_action)
                        temp_value = q[new_state][free_action]
                        if temp_value> max_new_state:
                            max_new_state = temp_value
                    q[current_state][action1]= current_value +  (aplha*(transition_reward+ 
                    (discount_factor*(max_new_state) - current_value)))
                    memory_track.append("win")
                    game_over = True;
                elif game_board.winner =="⭕":
                    transition_reward = game_board.loser_reward
                    action1 = tuple(action_player1)
                    q = check_Q(q,current_state,action1)
                    current_value = q[current_state][action1]
                    free_positions = game_board.free_space()
                    max_new_state = 0 
                    #Find the max of the next state for maxQ(St+1,A)
                    for position in free_positions:
                        free_action = tuple(position)
                        q = check_Q(q,new_state,free_action)
                        temp_value = q[new_state][free_action]
                        if temp_value> max_new_state:
                            max_new_state = temp_value
                    q[current_state][action1]= current_value +  (aplha*(transition_reward+ 
                    (discount_factor*(max_new_state) - current_value)))
                    memory_track.append("lose")
                    game_over = True;
                #Draw
                else:
                    transition_reward = game_board.reward 
                    action1 = tuple(action_player1)
                    q = check_Q(q,current_state,action1)
                    current_value = q[current_state][action1]
                    free_positions = game_board.free_space()
                    max_new_state = 0 
                    #Find the max of the next state for maxQ(St+1,A)
                    for position in free_positions:
                        free_action = tuple(position)
                        q = check_Q(q,new_state,free_action)
                        temp_value = q[new_state][free_action]
                        if temp_value> max_new_state:
                            max_new_state = temp_value
                    q[current_state][action1]= current_value +  (aplha*(transition_reward+ 
                    (discount_factor*(max_new_state) - current_value)))
                    memory_track.append("draw")
                    game_over = True;
            else:
                """
                Game is not over yet. It is the Agent's turn to take the next action and update the board
                """
                game_player1= game_board.player1
                action_player1 = greedy_action(game_board,q)
                current_state = new_state
                game_board.update_board(action_player1,game_player1)
                new_state = tuple(game_board.board.reshape(1,-1)[0])
                game_status = game_board.check_game_status()
                #Check the current status of the game after the agent taken a new action
                if game_status: 
                    #Agent wins update q table
                    if game_board.winner == "❌":
                        transition_reward = game_board.reward 
                        action1 = tuple(action_player1)
                        q = check_Q(q,current_state,action1)
                        current_value = q[current_state][action1]
                        free_positions = game_board.free_space()
                        max_new_state = 0 
                        #Find the max of the next state for maxQ(St+1,A)
                        for position in free_positions:
                            free_action = tuple(position)
                            q = check_Q(q,new_state,free_action)
                            temp_value = q[new_state][free_action]
                            if temp_value> max_new_state:
                                max_new_state = temp_value
                        q[current_state][action1]= current_value +  (aplha*(transition_reward+ 
                        (discount_factor*(max_new_state) - current_value)))
                        memory_track.append("win")
                        game_over = True;
                    #opponent wins update q table
                    elif game_board.winner =="⭕":
                        transition_reward = game_board.loser_reward
                        action1 = tuple(action_player1)
                        q = check_Q(q,current_state,action1)
                        current_value = q[current_state][action1]
                        free_positions = game_board.free_space()
                        max_new_state = 0 
                        #Find the max of the next state for maxQ(St+1,A)
                        for position in free_positions:
                            free_action = tuple(position)
                            q = check_Q(q,new_state,free_action)
                            temp_value = q[new_state][free_action]
                            if temp_value> max_new_state:
                                max_new_state = temp_value
                        q[current_state][action1]= current_value +  (aplha*(transition_reward+ 
                        (discount_factor*(max_new_state) - current_value)))
                        memory_track.append("lose")
                        game_over = True;
                    #Draw update q table
                    else:
                        transition_reward = game_board.reward 
                        action1 = tuple(action_player1)
                        q = check_Q(q,current_state,action1)
                        current_value = q[current_state][action1]
                        free_positions = game_board.free_space()
                        max_new_state = 0 
                        #Find the max of the next state for maxQ(St+1,A)
                        for position in free_positions:
                            free_action = tuple(position)
                            q = check_Q(q,new_state,free_action)
                            temp_value = q[new_state][free_action]
                            if temp_value> max_new_state:
                                max_new_state = temp_value
                        q[current_state][action1]= current_value +  (aplha*(transition_reward+ 
                        (discount_factor*(max_new_state) - current_value)))
                        memory_track.append("draw")
                        game_over = True;
        print(f"******************* Right now is on episode: {eps}")
        Winning_streak = 150
        mem_size = len(memory_track)
        """
        When the size of the memory is equal to 120:
        We will check if the whole sequences of that memorytrack is all wins
        Compute the percentage of total wins over Winning streak
        if the percentage is equal to 1 then prints out it's converges on that number of eps with optimal solution
        """
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
                print(memory_track)
                break
            else:
                pass
    print(memory_track)
    return q

"""
Picking the best action at the current state
"""
def max_action(game_board,q):
    free_positions = game_board.free_space()
    currentState = tuple(game_board.board.reshape(1,-1)[0])

        #Check all actions to find the best action leads to the best reward
    max_value = -np.inf
    for position in free_positions:
        free_action = tuple(position)
        q = check_Q(q,currentState,free_action)
        temp_value = q[currentState][free_action]
        if temp_value > max_value:
            max_value = temp_value
            action = position
    return action


#Using the q table play it againist a human player
def run_game(q_table,game_board):
    continue_game = True
    while continue_game != False:
        game_over= False
        print("You are player: ⭕")
        print("Computer is: ❌")
        game_board.reset_board()
        action_player1 = 1,1
        print(f"Computer taken: {action_player1}")
        print(game_board.print_board())
        print('\n\n')
        while game_over != True:
            game_player2= game_board.player2
            print(f"Here are the free positions in the game: {game_board.free_space()}")
            input_row = int(input("Please enter the desire row: "))
            input_col = int(input("Please enter the desire column: "))
            action_player2 = (input_row,input_col)
            game_board.update_board(action_player2,game_player2)
            game_board.print_board()
            print('\n\n')
            game_status = game_board.check_game_status()
            if game_status: 
                #Our player 1 trainee wins
                if game_board.winner == "❌":
                    game_over = True;
                #Oppoent wins
                elif game_board.winner =="⭕":
                    game_over = True;
                #Draw
                else:
                    game_over = True;
            else:
                game_player1= game_board.player1
                action_player1 = max_action(game_board,q_table)
                game_board.update_board(action_player1,game_player1)
                print(f"Computer taken: {action_player1}")
                game_board.print_board()
                print('\n\n')
                game_status = game_board.check_game_status()
                if game_status:
                    #Computer wins 
                    if game_board.winner == "❌":
                        game_over = True;
                    #Oppoent wins
                    elif game_board.winner =="⭕":
                        game_over = True;
                    #Draw
                    else:
                        game_over = True;
        print(f"Winner is : {game_board.winner}")
        input_play = input("Do you want to play again? 'yes' or 'no': ")
        if input_play == 'no':
            continue_game = False


#save the q table policy
def savePolicy(q_table):
    fw  = open("New_Trainner1",'wb')
    pickle.dump(q_table,fw)
    fw.close()


#load up the q table file 
def loadPolicy(file):
    fr = open(file,'rb')
    return pickle.load(fr)

#The main driver 
if __name__ == '__main__':
    #Setup the tic taco Toe game
    game_setup = TicTacoToe()

    """
    uncomment this Q_learning if you wanted to train the player1 from scratch but remember to uncomment the loadpolicy
    """
    # q_table = Q_learning(game_setup,500000)

    # savePolicy(q_table)

    """
    Can uncomment out to load up the "New_Trainner" policy. Therefore, also have to comcment out the Q_learning and savePolicy 
    """
    q_table = loadPolicy("New_Trainner1")
    print('\n\n')
    human_board = TicTacoToe()
    run_game(q_table,human_board)
    # print(q_table)
