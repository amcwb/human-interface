import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.cluster_detection import ClusterDetection
from . import colors

cd = ClusterDetection()

cam_ax = plt.subplot(1, 1, 1)
cam_im = cam_ax.imshow(cd._frame)


plt.ion()
boxes = []
lines = []
try:
    for key_rect in cd.find_key_points([
        ((100, 150, 0), (140, 255, 255)), # blue
        ((0, 58, 50), (30, 255, 255)), # skin
        ((160, 100, 20), (179, 255, 255))  # red
    ]):
        # Destroy boxes and lines from old render
        for box in boxes:
            box.remove()
        boxes = []

        for line in lines:
            line.remove()
        lines = []
        
        for k, key in enumerate(key_rect.keys()):
            cluster_data = key_rect[key]

            # Draw rect around cluster found
            box = Rectangle(
                (cluster_data[0], cluster_data[1]),
                (cluster_data[2] - cluster_data[0]),
                (cluster_data[3] - cluster_data[1]),
                fill=False,
                color=colors[k],
                alpha=1,
            )
            
            # Make a box
            boxes.append(box)
            cam_ax.add_patch(box)

        # Draw lines
        for p1, p2 in zip(list(key_rect.values()), list(key_rect.values())[1:]):
            if all(p == 0 for p in p1) or all(p == 0 for p in p2):
                continue
            
            center = lambda rect: [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2]

            # Draw line
            p1_center = center(p1)
            p2_center = center(p2)
            lines.extend(plt.plot(
                [p1_center[0], p2_center[0]],
                [p1_center[1], p2_center[1]],
                color='white'
            ))

        # Set backogrund image
        cam_im.set_data(cv2.cvtColor(cd._frame, cv2.COLOR_HSV2RGB))

        plt.pause(1/60)
except KeyboardInterrupt:
    pass
finally:
    cd._release_camera()