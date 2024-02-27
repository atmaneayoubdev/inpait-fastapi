import cv2
from utils import get_bounding_box_coordinates, draw_bounding_box


#  Get the bounding box coordinates:
top_left_x, top_left_y, bottom_right_x, bottom_right_y = get_bounding_box_coordinates(
    "https://images.bayut.com/thumbnails/595975711-800x600.webp")

# Draw the bounding box on the image
draw_bounding_box("https://images.bayut.com/thumbnails/595975711-800x600.webp",
                  top_left_x, top_left_y, bottom_right_x, bottom_right_y)
# draw_bounding_box("https://images.bayut.com/thumbnails/617797755-800x600.webp",
#                   0.6134868421052632, 0.5509868421052632, 0.3832236842105263, 0.4407894736842105)
