import heapq
import copy
from tokenize import ContStr
import numpy as np

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance=0
    numpy_state=np.reshape(from_state,(3,3))
    numpy_desired=np.reshape(to_state,(3,3))
    for i in range(3):
        for j in range(3):
            if numpy_state[i][j]==0:
                continue
      
            x=int(np.where(numpy_desired == numpy_state[i][j])[0])
            y=int(np.where(numpy_desired == numpy_state[i][j])[1])
        
            distance+=abs(i-x) +abs(j-y)
 
    return distance




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states=list()
    right_list=[1,2,4,5,7,8]
    left_list=[0,1,3,4,6,7]
    up_list=[0,1,2,3,4,5]
    down_list=[3,4,5,6,7,8]
    
    for i in range(len(state)):
        right=i+1
        left=i-1
        up=i-3
        down=i+3
        
        if right in right_list:
            if(state[right]==0):
                temp_state=copy.deepcopy(state)
                temp_state[right]=state[i]
                temp_state[i]=0
                if temp_state not in succ_states and temp_state!=state:
                    succ_states.append(temp_state)
        if left in left_list:
            if(state[left]==0):
                temp_state=copy.deepcopy(state)
                temp_state[left]=state[i]
                temp_state[i]=0
                if temp_state not in succ_states and temp_state!=state:
                    succ_states.append(temp_state)
        if up in up_list:
            if(state[up]==0):
                temp_state=copy.deepcopy(state)
                temp_state[up]=state[i]
                temp_state[i]=0
                if temp_state not in succ_states and temp_state!=state:
                    succ_states.append(temp_state)
        if down in down_list:
            if(state[down]==0):
                temp_state=copy.deepcopy(state)
                temp_state[down]=state[i]
                temp_state[i]=0
                if temp_state not in succ_states and temp_state!=state:
                    succ_states.append(temp_state)
                
        
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    open=list()
    visited=set()
    g=0
    parent_index=-1
    max_length=0 
    traceback=list()
    dict_index=0
    
    h=get_manhattan_distance(state)
    heapq.heappush(open,(g+h,state,(g,h,parent_index),0))
    visited.add(tuple(state))
    
    while open:
        max_length=max(max_length,len(open))
        n=heapq.heappop(open)

        if n[1]== [1,2,3,4,5,6,7,0,0]:
            break
        successors=get_succ(n[1])
        g=n[2][0]+1
        parent_index= n[2][2]+1
        dict_index+=1
        h=get_manhattan_distance(n[1])
        
        
        for successor in successors: 
            if tuple(successor) not in visited :
                h=get_manhattan_distance(successor)
                heapq.heappush(open,(g+h,successor,(g,h,parent_index),len(traceback)))  
                visited.add(tuple(successor))
                
            traceback.append(n)

        
    trace=list()
    parent=len(traceback)-1
    while(True):
        trace.insert(0,(n[1],n[2][1]))
        
        n=traceback[n[3]]  
        if parent == -1:
            break
        parent=n[2][2]
   
    moves=0
    for move in trace:
        print(move[0], "h={} moves: {}".format(move[1],moves))
        moves += 1 
    print("Max queue length: "+str(max_length))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    
    solve([4,3,0,5,1,6,7,2,0])    
