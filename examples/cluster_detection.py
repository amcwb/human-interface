import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.cluster_detection import ClusterDetection
from . import colors

"""
Test to detect red, blue and human skin
"""

cd = ClusterDetection()

# Initialize plot
cam_ax = plt.subplot(1, 1, 1)
cam_im = cam_ax.imshow(cd._frame)

plt.ion()
boxes = []
try:
    for filtered_data, rect_data in cd.feed_cluster_rect_data([
        ((0, 58, 50), (30, 255, 255)), # human skin
        ((100, 150, 0), (140, 255, 255)), # blue
        ((160, 100, 20), (179, 255, 255))  # red
    ]):
        # Remove old boxes
        for box in boxes:
            box.remove()
        boxes = []
        
        # Iterate through colors and clusters
        for k, key in enumerate(rect_data.keys()):
            for cluster in rect_data[key]:
                # Draw rect around cluster
                box = Rectangle(
                    (cluster[0], cluster[1]),
                    cluster[2] - cluster[0],
                    cluster[3] - cluster[1],
                    fill=False,
                    color=colors[k],
                    alpha=1,
                )

                boxes.append(box)
                cam_ax.add_patch(box)

        # Set background image
        cam_im.set_data(cv2.cvtColor(cd._frame, cv2.COLOR_HSV2RGB))

        plt.pause(1/60)
except KeyboardInterrupt:
    pass
finally:
    cd._release_camera()