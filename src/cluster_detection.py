from typing import List, Tuple
import numpy as np

from .detection import ColorBoundary, Detection
from skimage.measure import label


class ClusterDetection(Detection):
    def find_clusters(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Finds rects for clusters containing all pixels of the color (this
        function takes a filtered frame!).

        This returns None if no colored pixels were found.

        :param frame: The filtered frame to create a rect for
        :type frame: np.ndarray
        :return: The coordinates of the corners of the rect, optional
        :rtype: Tuple[int, int, int, int]
        """
        # Normalize frame values (to avoid false seperate groups)
        frame[frame != 0] = 1

        # Label
        labelled_frame = label(frame)
        
        # Find number of clusters
        num_clusters = np.amax(labelled_frame)
        clusters = []

        # Iterate through clusters
        for cluster in range(1, num_clusters + 1):
            filtered_cluster = labelled_frame.copy()
            filtered_cluster[filtered_cluster!=cluster] = 0
            clusters.append(self.find_basic_rect(filtered_cluster))

        return clusters

    def feed_cluster_rect_data(self, color_boundaries: List[ColorBoundary]):
        """
        Feed back coordinates

        :param color_boundaries: Color boundaries to watch for
        :type color_boundaries: List[ColorBoundary]
        """
        self._acquire_camera()
        try:
            while True:
                filtered_frames = {
                    color_boundary: None
                    for color_boundary in color_boundaries
                }

                rect_results = {
                    color_boundary: []
                    for color_boundary in color_boundaries
                }

                # Get data
                data = self.grab_frame()

                # Filter colors
                for color_boundary in color_boundaries:
                    filtered_frames[color_boundary] = self.filter_color(data, color_boundary)
                
                    # Find basic rects
                    rect_results[color_boundary] = self.find_clusters(filtered_frames[color_boundary]) or []
                
                yield filtered_frames, rect_results
        finally:
            self._release_camera()

    @staticmethod
    def _calculate_area(rect: Tuple[int, int, int, int]):
        """
        Calculate area of rect, assuming structure

            tlx, tly, brx, bry

        :param rect: The rect
        :type rect: Tuple[int, int, int, int]
        """
        return (rect[2] - rect[0]) * (rect[3] - rect[1])

    @staticmethod
    def _normalize(rect: Tuple[int, int, int, int], size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """
        Normalize rect for frame

        :param rect: Rect to normalize
        :type rect: Tuple[int, int, int, int]
        :param size: The frame size, height then width
        :type size: Tuple[int, int]
        :return: Tuple[float, float, float, float]
        :rtype: Normalized rect between 0 and 1.
        """
        return (
            rect[0] / size[1],
            rect[1] / size[0],
            rect[2] / size[1],
            rect[3] / size[0]
        )

    def find_key_points(self, color_boundaries: List[ColorBoundary]):
        """
        Finds "key" points from the color boundaries.

        :param color_boundaries: Color boundaries to watch for
        :type color_boundaries: List[ColorBoundary]
        """
        for _, rect_data in self.feed_cluster_rect_data(color_boundaries):
            # Find biggest rect per color
            color_areas = {
                color_boundary: 0
                for color_boundary in color_boundaries
            }

            color_rects = {
                color_boundary: (0, 0, 0, 0)
                for color_boundary in color_boundaries
            }

            for key in rect_data.keys():
                for cluster in rect_data[key]:
                    cluster_area = self._calculate_area(cluster)
                    if cluster_area > color_areas[key]:
                        color_areas[key] = cluster_area
                        color_rects[key] = cluster
            
            yield color_rects


    def find_key_points_normalized(self, color_boundaries: List[ColorBoundary]):
        """
        Finds "key" points from the color boundaries normalized between 0 and
        1. Wraps `find_key_points` and `_normalize`.

        :param color_boundaries: Color boundaries to watch for
        :type color_boundaries: List[ColorBoundary]
        """
        for color_rects in self.find_key_points(color_boundaries):
            for key, value in color_rects.items():
                color_rects[key] = self._normalize(value, self._frame.shape)
            
            yield color_rects
