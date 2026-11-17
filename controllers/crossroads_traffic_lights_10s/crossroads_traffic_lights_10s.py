"""Crossroads traffic light: every light changes every 10 s (green 10 s, orange 2 s, red 12 s).
Lamp numbering as in Webots' crossroads_traffic_lights.c (0,1 / 6,7 straight+right and 2 / 8 left
arrow for approaches 1+3; 3,4 / 9,10 and 5 / 11 for approaches 2+4).
Every lamp not green/orange in the current phase is red. Used via CrossRoadsTrafficLight {
controller "crossroads_traffic_lights_10s" } in our worlds.
"""
from controller import Robot

GREEN, ORANGE = 10.0, 2.0
# Two phases so every light changes every 10 s (+2 s orange): approaches 1+3 (straight, right and
# left arrow), then approaches 2+4. A red therefore never lasts more than 12 s.
PHASES = [(0, 1, 2, 6, 7, 8), (3, 4, 5, 9, 10, 11)]

robot = Robot()
step = 64
leds = {c: [robot.getDevice(f'{c} light {i}') for i in range(12)] for c in ('red', 'orange', 'green')}


def show(active, colour):
    for i in range(12):
        on = i in active
        leds['green'][i].set(1 if on and colour == 'green' else 0)
        leds['orange'][i].set(1 if on and colour == 'orange' else 0)
        leds['red'][i].set(0 if on else 1)


t, last = 0.0, None
cycle = len(PHASES) * (GREEN + ORANGE)
while robot.step(step) != -1:
    t = (t + step / 1000.0) % cycle
    k = int(t // (GREEN + ORANGE))
    colour = 'green' if t - k * (GREEN + ORANGE) < GREEN else 'orange'
    if (k, colour) != last:
        last = (k, colour)
        show(PHASES[k], colour)
