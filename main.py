import json
import math
import random
import sys

from matplotlib import cm
import numpy as np
from PIL import Image


DEFAULT_CONFIG = {
  "color": False,
  "width": 300,
  "height": 300,
  "density": 0.1,
  "maxCircleRadius": 0.2,
  "minCircleRadius": 0.05
}

class Circle:
    def __init__(self, x: int, y: int, radius: int):
        self.x = x
        self.y = y
        self.radius = radius

    def __str__(self) -> str:
        return f"x={self.x}, y={self.y}, radius={self.radius}"

class Sculptor:
    def load_config(self):
        self.config = DEFAULT_CONFIG
        return self.config

    def create_field(self) -> list[list[float]]:
        self.field = [
            [0] * self.config['width']
            for i in range(self.config['height'])
        ]

        return self.field

    def apply_circle(self, circle: Circle) -> None:
        for x in range(max(0, circle.x - circle.radius), min(self.config['width'], circle.x + circle.radius)):
            for y in range(max(0, circle.y - circle.radius), min(self.config['height'], circle.y + circle.radius)):
                distance = math.sqrt((circle.x - x) ** 2 + (circle.y - y) ** 2)
                if distance < circle.radius:
                    self.field[y][x] += 1 - distance / circle.radius 

    def place_circles(self):
        print()
        min_side = min(self.config['width'], self.config['height'])
        num_circles = min_side * self.config['density']
        min_radius = int(min_side * self.config['minCircleRadius'])
        max_radius = int(min_side * self.config['maxCircleRadius'])

        self.circles = []
        for i in range(int(num_circles)):
            circle = Circle(
                x=random.randint(0, self.config['width']),
                y=random.randint(0, self.config['height']),
                radius=random.randrange(min_radius, max_radius)
            )
            self.circles.append(circle)
            self.apply_circle(circle)

            print("Circle:", circle)
        
    def generate(self):
        self.load_config()
        self.create_field()
        self.place_circles()
        return self.field
    
    def print(self):
        print()

        if self.config['color'] == True:
            array = np.uint8(cm.gist_earth(self.field) * 255)
        else:
            pixels = [[min(255, int(p * 255)) for p in row] for row in self.field]
            array = np.array(pixels, dtype=np.uint8)
            
        image = Image.fromarray(array)
        image.save('output.png')
        print()

if __name__ == '__main__':
    print("Running...")
    sculptor = Sculptor()
    sculptor.generate()
    sculptor.print()
    print("Done!")