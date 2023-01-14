desiredNumber = input("How many Fibonacci numbers would you like to generate? ")

f0 = 1
f1 = 1
FibSeq = [f0, f1]

i = 2
while i < int(desiredNumber):
    f2 = f1 + f0
    FibSeq.append(f2)
    
    f0 = f1
    f1 = f2
    i += 1
    
print("The Fibonacci Sequence is: " + str(FibSeq))
