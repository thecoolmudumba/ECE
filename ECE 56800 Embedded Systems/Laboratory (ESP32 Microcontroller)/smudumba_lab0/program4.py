Dictionary = {"Person A" : "01/01/2000","Person B" : "02/01/1990","Person C" : "03/01/1800"}

print("Welcome to the birthday dictionary. We know the birthdays of: ")
for k, v in Dictionary.items():
    print(k)
    
lookUp = input("Whose birthday do you want to look up?\n")

print(lookUp + "'s birthday is " + Dictionary[lookUp])


