"""
ECE 56800 Lab02 ESP CLIENT
Author: Sai V Mudumba
Date Modified: March 25, 2022
"""

import esp32
from machine import Timer, Pin
import machine
import network

try:
    import usocket as socket
except:
    import socket

# Connect to Wi-Fi
# https://docs.micropython.org/en/latest/esp32/quickref.html?highlight=dht
sta_if = network.WLAN(network.STA_IF)
if not sta_if.isconnected(): # if Not Connected
    print("Connecting to the network...")
    sta_if.active(True)
    SSID = "<INSERT YOUR SSID HERE>" # e.g., "iPhone" or "Apartment300"
    Password = "<INSERT YOUR NETWORK PASSWORD HERE>" 
    sta_if.connect(SSID, Password)
    while not sta_if.isconnected():
        pass
print("Network Configuration: ", sta_if.ifconfig())

def handleTimerInterrupt(timer):
    # https://docs.micropython.org/en/latest/esp32/quickref.html?highlight=dht
    hall_sensor = esp32.hall_sensor()
    raw_temperature = esp32.raw_temperature()
    print(f'Onboard Temperature: {raw_temperature} deg F; Hall Sensor: {hall_sensor}')
    
    # https://docs.micropython.org/en/latest/esp8266/tutorial/network_tcp.html
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = socket.getaddrinfo("api.thingspeak.com", 80)[0][-1]
    s.connect(addr)
    s.send(f"GET https://api.thingspeak.com/update?api_key=POUKVSBOAOJA3J5P&field1={raw_temperature}&field2={hall_sensor} HTTP/1.0\r\n\r\n")
    while True:
        data = s.recv(100)
        if data:
            machine.idle()
        else:
            break
    s.close()
    
# initialize the timer
timer = machine.Timer(0)

# every 30 seconds (30,000 ms), print out the date and time
timer.init(period=30000, mode=Timer.PERIODIC, callback=handleTimerInterrupt)

    





