import copy
import numpy as np
def degenate(allocation,num_sources,num_destinations,cost_matrix) :
    min_cost = np.inf
    min_i, min_j = -1, -1
    for i in range(num_sources):
        for j in range(num_destinations):
            if allocation[i][j] == 0 and cost_matrix[i][j] < min_cost:
                # Check if allocating 0.001 to this cell creates dependencies
                    min_cost = cost_matrix[i][j]
                    min_i, min_j = i, j

    if min_i != -1 and min_j != -1:
        # Allocate 0.001 to the cell with the least cost that doesn't create dependencies
        allocation[min_i][min_j] = 0.1
        print(f"Allocated 0.001 to cell ({min_i + 1}, {min_j + 1}) to resolve degeneracy.")
    print(allocation)
    count=0
    for i in range(rows):  
        for j in range(cols): 
            if allocation[i][j]!=0:
                count=count+1
    if count < num_sources+num_destinations-1:
        print("\nDegenerate\n")
        degenate(allocation,num_sources,num_destinations,cost_matrix)
    else:
        print("\nNon-Degenerate\n")
        
        
def least_cost_method(cost_matrix, supply, demand):
    num_sources, num_destinations = cost_matrix.shape
    allocation = np.zeros((num_sources, num_destinations))
    
    while np.sum(supply) > 0 and np.sum(demand) > 0:
        min_cost_indices = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
        source_idx, dest_idx = min_cost_indices
        
        quantity = min(supply[source_idx], demand[dest_idx])
        
        allocation[source_idx, dest_idx] = quantity
        supply[source_idx] -= quantity
        demand[dest_idx] -= quantity
        
        cost_matrix[source_idx, dest_idx] = np.inf
    
    return allocation

# Input data

num_sources = int(input("Enter the number of supply: "))
num_destinations = int(input("Enter the number of demand: "))

print("Enter the cost matrix:")
cost_matrix = np.zeros((num_sources, num_destinations))
for i in range(num_sources):
    cost_matrix[i] = list(map(int, input("Enter cost row-wise").split()))

print(f'cost_matrix = \n{cost_matrix}')

supply = list(map(int, input("Enter the supply values ").split()))
demand = list(map(int, input("Enter the demand values ").split()))
print(f'supply values = \n{supply}')
print(f'demand values = \n{demand}')

# Balancing the problem
total_supply = np.sum(supply)
total_demand = np.sum(demand)
if total_supply > total_demand:
    demand.append(total_supply - total_demand)
    cost_matrix = np.vstack((cost_matrix, np.zeros(num_destinations)))
    print(f'Given Transportation problem is not Balanced')
    print(f'By adding dummy column')
    print(f'We make Transportation problem Balanced')
elif total_demand > total_supply:
    supply.append(total_demand - total_supply)
    cost_matrix = np.hstack((cost_matrix, np.zeros((num_sources, 1))))
    print(f'Given Transportation problem is not Balanced')
    print(f'By adding dummy row')
    print(f'We make Transportation problem Balanced')
else:
    print(f'Given Transportation problem is Balanced')

cost_matrix2=copy.deepcopy(cost_matrix)
rows = len(supply)
cols = len(demand)
max_ele=np.max(cost_matrix)
for i in range(rows):
    for j in range(cols):
        cost_matrix2[i][j]=max_ele - cost_matrix[i][j]
        
d=int(input("Enter 1 for min 2 for max"))
if d==1:
    allocation = least_cost_method(np.copy(cost_matrix), np.copy(supply), np.copy(demand))
else:
    allocation = least_cost_method(np.copy(cost_matrix2), np.copy(supply), np.copy(demand))
    
print("Initial  basic feasible solution using Least Cost Method:\n", allocation)
rows = len(supply)
cols = len(demand)
for i in range(rows):
    for j in range(cols):
        if allocation[i][j] > 0:
            print(allocation[i][j],f'units from S{i+1} to D{j+1}') 
            
rows = len(supply)
cols = len(demand)
totat_cost=0
for i in range(rows):  
    for j in range(cols): 
        totat_cost=totat_cost+cost_matrix[i][j]*allocation[i][j]
count=0     
for i in range(rows):  
    for j in range(cols): 
        if allocation[i][j]!=0:
            count=count+1
print('\nn+m-1 = ', (num_sources+num_destinations-1))
print(f'Transportation cost = {totat_cost}')
if count < num_sources+num_destinations-1:
    print("\nDegenerate\n")
    if d==1:
        degenate(allocation,num_sources,num_destinations,cost_matrix)
    else:
        degenate(allocation,num_sources,num_destinations,cost_matrix2)