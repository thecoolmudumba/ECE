import random

randomNumber = random.randint(0,10)
guessLimit = 3
# print(randomNumber)

for c in range(3):
    guess = input("Enter your guess: ")
    
    if int(guess) == int(randomNumber):
        print("You win!")
        break
    elif c >= guessLimit-1:
        print("You lose!")
        break