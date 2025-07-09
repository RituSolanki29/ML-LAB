# %%
# Counting the number of pairs with sum = 10
def countPair():            
    count =0
    arr = [2,7,4,1,3,6]
    n = len(arr)
    for i in range(n):
        for j in range(i+1,n):
            if (arr[i]+arr[j] == 10):
                count = count+1
    
    return count

def main():
    x = countPair()
    print("No. of pairs equal to 10 ",x)

main()

#%%
# Calulating the range of the given input List and also error message if length is less then 3
def retRange(arr):
    n = len(arr)
    if(n <3):
        return "Range determination is not possible "
    else:
        Min = min(arr)
        Max = max(arr)
        return (Min,Max)

def main():
    n = input("Enter the elements of the list:")
    arr = list(map(int,n.strip().split()))
    x = retRange(arr)
    print("Range of List is",x)

main()
        
# %%
# returning the matrix multiplied by m times
def matrix(A,m):
    n = len(A)
    def matrix_matrix(p,q):
        result = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += p[i][j]*q[i][j];
        return result
    
    result = A
    for i in range(m-1):
        result = matrix_matrix(result, A)
    return result
    
def main():
    n = int(input("Enter the size of the matrix:"))
    print("Enter the elemetns of the matrix:")
    A = []
    for i in range(n):
        arr = list(map(int,input().split()))
        A.append(arr)
    
    m = int(input("Enter the positive number:"))
    if(m<1):
        print("not positive")

    x = matrix(A,m)

    for r in x:
        print(r)

main()

#%%
# Returning the highest occurence char in a string with frequncy
def highOccurnce(s):
    count= {}

    for i in s:
        if i != ' ':
            count[i] = count.get(i,0)+1
    
    character = max(count, key = count.get)
    max_count = count[character]

    return character, max_count
        
def main():
    s = input("Enter the String:")
    char, count = highOccurnce(s)
    print("Highest occurence charcter is'{char}' with count {count}")

main()

#%%
# returning the mode,mean and median of the randomly generated 25 number in range of 1 to 10
import random
import statistics

random_List = [random.randint(1,10) for _ in range(25)]
print("List",random_List)

value_mean = statistics.mean(random_List)
value_median = statistics.median(random_List)
value_mode = statistics.mode(random_List)

print("Mean",value_mean)
print("Median", value_median)
print("Mode", value_mode)
