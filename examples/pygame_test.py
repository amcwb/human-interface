import pygame
from pygame import color
from src.cluster_detection import ClusterDetection

from . import colors

# Get center of Color Detection rect
def center(rect):
    return [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2]

# Dimensions
width = 900
height = 800

# Initailize pygame
pygame.init()
screen = pygame.display.set_mode((width, height))

# Initialize color detections
cd = ClusterDetection()

# Color range definitions
red_range = ((160, 100, 20), (179, 255, 255))
blue_range = ((100, 150, 0), (140, 255, 255))
frames = cd.find_key_points_normalized([
    blue_range, # blue
    ((0, 58, 50), (30, 255, 255)), # human skin
    red_range  # red
])

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit

    # Fill screen
    screen.fill((0, 0, 0))

    # Iterate through rects and draw lines
    color_rects = next(frames)
    rects = list(color_rects.values())
    for k, (r1, r2) in enumerate(zip(rects, rects[1:])):
        if all(p == 0 for p in r1) or all(p == 0 for p in r2):
            continue

        
        # Get centers and draw lines
        r1_center = center(r1)
        r2_center = center(r2)
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (r1_center[0] * width, r1_center[1] * height),
            (r2_center[0] * width, r2_center[1] * height)
        )
    
    # Draw circles and surroundings rects
    for k, (key, value) in enumerate(color_rects.items()):
        if all(p == 0 for p in value):
            continue
        
        color_center = center(value)
        pygame.draw.circle(screen, colors[k], (color_center[0] * width, color_center[1] * height), 25)
        pygame.draw.rect(screen, colors[k], (
            value[0] * width,
            value[1] * height,
            (value[2]-value[0]) * width,
            (value[3]-value[1]) * height,
        ), width=3)

    # Show to screen
    pygame.display.flip()