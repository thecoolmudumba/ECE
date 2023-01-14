"""
COURSE: ECE 56800 EMBEDDED SYSTEMS
LAB 1: Exercising Peripherals (Interrupts, Timers, RTC, GPIO, PWM)
AUTHOR: Sai V. Mudumba
DATE: March 6, 2022
"""

from machine import Timer
import machine
from machine import Pin, PWM

# USER INPUTS
Year = int(input("Year? "))
Month = int(input("Month? "))
Day = int(input("Day? "))
Weekday = int(input("Weekday? "))
Hour = int(input("Hour? "))
Minute = int(input("Minute? "))
Second = int(input("Second? "))
Microsecond = int(input("Microsecond? "))

# INITIALIZE THE RTC TIMER FROM USER INPUTS
# https://mpython.readthedocs.io/en/master/library/micropython/machine/machine.RTC.html
rtc = machine.RTC()
rtc.datetime((Year,Month,Day,Weekday,Hour,Minute,Second,Microsecond))

# PRINT THE CURRENT RTC OUTPUT
# https://microcontrollerslab.com/micropython-timers-esp32-esp8266-generate-delay/
def handleTimerInterrupt(timer):
    global rtc
    rtc_call = rtc.datetime()
    Date = rtc_call[0:3]
    Time = rtc_call[4:8]
    print("Date: " + str(Date[1]) + "/" + str(Date[2]) + "/" + str(Date[0]) + " | Time: " + str(Time[0]) + ":" + str(Time[1]) + ":" + str(Time[2]) + ":" + str(Time[3]))

"""
handleButtonInterrupt() initiates a one shot timer of 200ms, after which it calls a debounce function
"""
# https://kaspars.net/blog/micropython-button-debounce
def handleButtonInterrupt(change):
    timer.init(mode=Timer.ONE_SHOT, period=200, callback=debounce)
    
"""
After 200ms of timer, debounce registers the button count once, which is then used to toggle led states
"""
def debounce(timer):
    global led, f, button_pressed_count, i
    #print(button_pressed_count)
    #print('pressed')
    
    led.freq(f[i])
    led.duty(256)
    
    if button_pressed_count % 2 == 0:
        i = 1
    elif button_pressed_count % 2 == 1:
        i = 0
    
    button_pressed_count += 1

        
# initialize the timer
timer = machine.Timer(0)

# initialize a list of PWM frequencies [Hz] to change when button is pressed
f = [5, 1]

# initialize toggle states
i = 0

# count the button presses
button_pressed_count = 0

# every 10 seconds (10,000 ms), print out the date and time
timer.init(period=10000, mode=Timer.PERIODIC, callback=handleTimerInterrupt)

# assign led to pin 13 on ESP32 board; define PWM to the led with freq and duty cycle
led = Pin(13, Pin.OUT)
led = PWM(led)
led.freq(30)
led.duty(0)

# assign the push button to pin 12 on ESP32 board
push_button = Pin(12,  Pin.IN, Pin.PULL_DOWN)

# when push button is pressed, it turns on the led, and off when pushed again
push_button.irq(handler = handleButtonInterrupt, trigger=Pin.IRQ_FALLING)
