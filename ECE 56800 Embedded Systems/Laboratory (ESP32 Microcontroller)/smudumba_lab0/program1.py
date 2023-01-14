from datetime import datetime

# https://www.w3schools.com/python/python_user_input.asp
name = input("What is your name? ")
age = input("How old are you? ")

# https://stackoverflow.com/questions/28189442/datetime-current-year-and-month-in-python
currentYear = datetime.now().year

# Calculate the year when the person will turn 100
# Using 2 assumptions: person birthday has or hasn't already past this year
age2Hundred = (100-int(age))+currentYear
print(name + " will turn 100 years old in either " + str(age2Hundred-1) + " or " + str(age2Hundred))
