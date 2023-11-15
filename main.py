import json
import random
import sys

DEFAULT_CONFIG = {
  "width": 30,
  "height": 30,
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
            self.field[circle.y][circle.x] = 1

            print("Circle:", circle)
        
    def generate(self):
        self.load_config()
        self.create_field()
        self.place_circles()
        return self.field
    
    def print(self):
        print()
        for row in self.field:
            print(row)
        print()

if __name__ == '__main__':
    print("Running...")
    sculptor = Sculptor()
    sculptor.generate()
    sculptor.print()
    print("Done!")