import numpy as np
import copy
def get_min_min2(array):
    m1,m2 = np.inf,np.inf
    for val in array:
        if val<=m1:
            m1,m2 = val,m1
        elif val<=m2:
            m2 = val
    return m1,m2

def get_max_penalty(cost, n_factories, n_centers):
    print(cost)
    row_penalty = []
    for r in range(n_factories):
        m1,m2 = get_min_min2(cost[r])
        if m1!=np.inf and m2!=np.inf:
            row_penalty.append((m2-m1, r))
        elif m1!=np.inf:
            row_penalty.append((m1,r))
    
    col_penalty = []
    for c in range(n_centers):
        col_m = []
        for i in range(n_factories):
            col_m.append(cost[i][c])
        m1,m2 = get_min_min2(col_m)
        if m1!=np.inf and m2!=np.inf:
            col_penalty.append((m2-m1, c))
        elif m1!=np.inf:
            col_penalty.append((m1,c))

    if not (row_penalty or col_penalty):
        return -1,-1,None    

   

    r_mp,c_mp = max(row_penalty),max(col_penalty)
    if r_mp>=c_mp:
        i = r_mp[1]
        mp = r_mp[0]
        j=0
        mn = cost[i][j]
        for c in range(n_centers):
            if cost[i][c]<mn:
                j = c
                mn = cost[i][c]
    else:
        j = c_mp[1]
        mp = c_mp[0]
        i=0
        mn = cost[i][j]
        for r in range(n_factories):
            if cost[r][j]<mn:
                i = r
                mn = cost[i][c]
    return i,j,mp

supply = []
demand = []
cost = []
sum_demand, sum_supply, total_cost, no_alloc = 0, 0, 0, 0

n_factories = int(input("Enter number of Factories : "))
n_centers = int(input("Enter number of Centers : "))

supply = list(map(int,input("Enter the supply  : ").strip().split()))[:n_factories]
demand = list(map(int,input("Enter the demand  : ").strip().split()))[:n_centers]


for i in range(n_factories):
    r = list(map(int,input("Enter cost row-wise").strip().split()))[:n_centers]
    cost.append(r)

vgb = copy.deepcopy(cost)
    
print(f'supply values = {supply}')
print(f'demand values = {demand}')

print(f'\ncost_matrix = {cost}\n')



sum_supply = sum(supply)
sum_demand = sum(demand)

if(sum_supply==sum_demand):
    print("\nGiven Transportation Problem is Balanced")
else:
    if sum_supply<sum_demand:
        cost=np.append(cost,np.array(np.zeros((1,n_centers))), axis=0)
        supply=np.append(supply,sum_demand-sum_supply)
        n_factories+=1
        print(f'Given Transportation problem is not Balanced')
        print(f'By adding dummy row')
        print(f'We make Transportation problem Balanced')
        print(f'\ncost_matrix after adding dummy row = {cost}\n')
    elif sum_supply>sum_demand:
        cost=np.concatenate((cost,np.zeros((n_factories,1))),axis=1)
        demand=np.append(demand,sum_supply-sum_demand)
        n_centers+=1
        print(f'Given Transportation problem is not Balanced')
        print(f'By adding dummy column')
        print(f'We make Transportation problem Balanced')
        print(f'\ncost_matrix after adding dummy column = {cost}\n')

allocation = []
for k in range(n_factories):
    temp = [0]*n_centers
    allocation.append(temp)

cost_matrix2=copy.deepcopy(cost)
rows = len(supply)
cols = len(demand)
max_ele=np.max(cost)
for i in range(rows):
    for j in range(cols):
        cost_matrix2[i][j]=max_ele - cost[i][j]
vgb2=copy.deepcopy(cost_matrix2)
d=int(input("Enter 1 for min 2 for max"))
if d==1:
    i,j,max_penalty = get_max_penalty(cost, n_factories, n_centers)
    while(max_penalty!=None):
       
        x = min(supply[i],demand[j])
        supply[i] = supply[i] - x
        demand[j] = demand[j] - x
        
        if x > 0 :
            allocation[i][j] = x
            no_alloc = no_alloc + 1  
            total_cost = total_cost + x*cost[i][j]
        
        if supply[i] == 0:
            for j in range(n_centers):
                cost[i][j] = np.inf
        if demand[j] == 0:
            for i in range (n_factories):
                cost[i][j] = np.inf

        i,j,max_penalty = get_max_penalty(cost, n_factories, n_centers)
else:
    i,j,max_penalty = get_max_penalty(cost_matrix2, n_factories, n_centers)
    
    while(max_penalty!=None):
    
        
        x = min(supply[i],demand[j])
        supply[i] = supply[i] - x
        demand[j] = demand[j] - x
        
        if x > 0 :
            allocation[i][j] = x
            no_alloc = no_alloc + 1  
            total_cost = total_cost + x*cost[i][j]
        
        if supply[i] == 0:
            for j in range(n_centers):
                cost_matrix2[i][j] = np.inf
        if demand[j] == 0:
            for i in range (n_factories):
                cost_matrix2[i][j] = np.inf

        i,j,max_penalty = get_max_penalty(cost_matrix2, n_factories, n_centers)

print("\nInitial  basic feasible solution using Vogel’s Approximation Method:\n",allocation)


print("\nTransportation Distribution:\n")
rows = len(supply)
cols = len(demand)
for i in range(rows):
    for j in range(cols):
        if allocation[i][j] > 0:
            print(allocation[i][j],f'units from S{i+1} to D{j+1}')  

print("\nTotal cost of transportation = ", total_cost)

print('\nNo. of Allocations = ', no_alloc)
print('\nn+m-1 = ', (n_factories+n_centers-1))

if no_alloc < n_factories+n_centers-1:
    print("\nDegenerate\n")
    if d==1:
        min_cost = np.inf
        min_i, min_j = -1, -1
        for i in range(n_factories):
            for j in range(n_centers):
                if allocation[i][j] == 0 and vgb[i][j] < min_cost:
                    # Check if allocating 0.001 to this cell creates dependencies
                        min_cost = vgb[i][j]
                        min_i, min_j = i, j

        if min_i != -1 and min_j != -1:
            # Allocate 0.001 to the cell with the least cost that doesn't create dependencies
            allocation[min_i][min_j] = 0.1
            print(f"Allocated 0.1 to cell ({min_i + 1}, {min_j + 1}) to resolve degeneracy.")
        print(allocation)
    else:
        min_cost = np.inf
        min_i, min_j = -1, -1
        for i in range(n_factories):
            for j in range(n_centers):
                if allocation[i][j] == 0 and vgb2[i][j] < min_cost:
                    # Check if allocating 0.001 to this cell creates dependencies
                        min_cost = vgb2[i][j]
                        min_i, min_j = i, j

        if min_i != -1 and min_j != -1:
            # Allocate 0.001 to the cell with the least cost that doesn't create dependencies
            allocation[min_i][min_j] = 0.1
            print(f"Allocated 0.1 to cell ({min_i + 1}, {min_j + 1}) to resolve degeneracy.")
        print(allocation)
        
else:
  print("\nNon-Degenerate\n")
