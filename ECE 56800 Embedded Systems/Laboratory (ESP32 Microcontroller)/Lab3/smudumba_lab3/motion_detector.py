from machine import Timer
import machine
from machine import I2C, Pin, PWM
from time import sleep
import network
import esp32
import mpu6050
import urequests as requests

try:
    import usocket as socket
except:
    import socket

led_board_green = Pin(13, Pin.OUT)
led_board_red = Pin(12, Pin.OUT)

#for i in range(1):
#    led_board_red.value(not led_board_red.value())
#    led_board_green.value(not led_board_green.value())

#    sleep(0.5)
        
# CONNECT TO WIFI
# https://docs.micropython.org/en/latest/esp32/quickref.html?highlight=dht
sta_if = network.WLAN(network.STA_IF)
if not sta_if.isconnected(): # if Not Connected
    print("Connecting to the network...")
    sta_if.active(True)
    SSID = "Apartment107"
    Password = "Newton98"
    sta_if.connect(SSID, Password)
    while not sta_if.isconnected():
        pass
print("Network Configuration: ", sta_if.ifconfig())

def handleTimerInterrupt(timer):
    global rtc
    rtc_call = rtc.datetime()
    Date = rtc_call[0:3]
    Time = rtc_call[4:8]
    #print("Date: " + str(Date[1]) + "/" + str(Date[2]) + "/" + str(Date[0]) + " | Time: " + str(Time[0]) + ":" + str(Time[1]) + ":" + str(Time[2]) + ":" + str(Time[3]))
    return (Time[2])

def handleDetectorInterrupt(timer):   
    # Check if Google Assistant Activated the Motion Detection Device in ThingSpeak
    print("I am checking to see if the last Google Assistant Command was ACTIVATE or DEACTIVATE...")
    
    response_API = requests.get('https://api.thingspeak.com/channels/1722153/feeds.json?api_key=LHK06NR3WATFWQJO&results=2')
    response_API = response_API.json()['feeds'][-1]['field1']
    print("\nThe Message sent was: "+str(response_API))
    return response_API

# INITIALIZE THE RTC TIMER FROM USER INPUTS
# https://mpython.readthedocs.io/en/master/library/micropython/machine/machine.RTC.html
rtc = machine.RTC()
rtc.datetime((2022,1,1,1,1,0,0,0))
# initialize the timer
timer = machine.Timer(0)

response_API = "deactivate"

while response_API == "deactivate":
    response_API = handleDetectorInterrupt(timer)
    sleep(1)
    
if response_API == "Activate" or response_API == "activate":
    # Turn on the GREEN LED light
    led_board_green.value(not led_board_green.value())
    
    i2c = I2C(scl=Pin(22), sda=Pin(23))
    accelerometer = mpu6050.accel(i2c)
    # Calibration Code
    i = 0
    totSamp = 10
    calibrateX = 0
    calibrateY = 0
    calibrateZ = 0
    while i < totSamp:
        sleep(0.1)
        accelerometer.get_values()
        print(accelerometer.get_values())
        calibrateX += (accelerometer.get_values()['AcX'])
        calibrateY += (accelerometer.get_values()['AcY'])
        calibrateZ += (accelerometer.get_values()['AcZ'])
        i += 1
       
    calibrateX = -calibrateX / totSamp
    calibrateY = -calibrateY / totSamp
    calibrateZ = 1 - calibrateZ / totSamp
    print(f'Calibrate Complete: X by {calibrateX}, Y by {calibrateY}, Z by {calibrateZ}')

    while True:
        sleep(1)
        accelDataXYZ = accelerometer.get_values()
        accelX = accelDataXYZ['AcX'] + calibrateX
        accelY = accelDataXYZ['AcY'] + calibrateY
        accelZ = accelDataXYZ['AcZ'] + calibrateZ
        
        # https://gpiocc.github.io/learn/micropython/esp/2020/06/06/martin-ku-send-notifications-from-esp32-to-telegram-with-ifttt.html
        if accelX > 1.5 or accelY > 1.5 or accelZ > 1.5:
            led_board_red.value(not led_board_red.value())
            webhooks_url = f"https://maker.ifttt.com/trigger/Motion_Detected/with/key/b4_DQsKrbeAZUPxel2v9SO?value1={accelX}&value2={accelY}&value3={accelZ}"
            try:
                r = requests.get(webhooks_url)
                print(r.text)
            except Exception as e:
                print(e)
            sleep(20)
        
        # Periodically check after 30s, Initiate handleDetectorInterrupt, which checks whether Activate or Deactivate 
        #timer.init(period=10000, mode=Timer.PERIODIC, callback=handleDetectorInterrupt)
        timeCounter = handleTimerInterrupt(timer)
        print(f'Seconds: {timeCounter}, Accl X: {accelX}, Accl Y: {accelY}, Accl Z: {accelZ}')
        if timeCounter % 30 == 0:
            # check the status of activate and deactivate
            response_API = handleDetectorInterrupt(timer)
            if response_API == "deactivate":
                led_board_green.value(not led_board_green.value())
                led_board_red.value(not led_board_red.value())
                break
    
    

