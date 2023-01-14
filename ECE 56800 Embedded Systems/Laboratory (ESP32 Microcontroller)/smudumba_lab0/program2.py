import random

# list size
n = 10
# create an empty list
A = []

# https://www.tutorialspoint.com/generating-random-number-list-in-python
for i in range(n):
    a = random.randint(0,100)
    A.append(a)
    
print("a = " + str(A))

number = input("Enter number (between 0 and 100): ")

A_new = []
for i in range(len(A)):
    if A[i] < int(number):
        A_new.append(A[i])

print("The new list is " + str(A_new))


    