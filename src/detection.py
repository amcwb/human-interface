# detection.py
#
# This file handles actually finding the data from the camera input and then
# producing virtual coordinates as outputs
from typing import List, Tuple

import cv2
import numpy as np


ColorBoundary = Tuple[Tuple[int, int, int], Tuple[int, int, int]]

class Detection:
    """
    This class handles collecting data from the camera and returning it.

    There are utility functions:
    
        Detection._acquire_camera()
        Detection._release_camera()

    To prevent hogging the camera, however this can be toggled.
    """
    def __init__(self, resize_width: int = 300, resize_height: int = 169):
        """
        Create a detection instance with colour boundaries
        """
        # Set resize data
        self.resize_width = resize_width
        self.resize_height = resize_height

        # Acquire first frame
        self._acquire_camera()
        self._frame = self.grab_frame()
        self._release_camera()

    def grab_frame(self, update_frame: bool = True) -> np.ndarray:
        """
        Get an RGB frame from the camera at time of execution and store it as
        the last frame.

        :param update_frame: Whether to update the local frame, defaults to True
        :type update_frame: bool, optional
        :return: The frame
        :rtype: np.ndarray
        """
        if self.cap is None:
            raise ValueError("Camera is not acquired. Run _acquire_camera first")
        
        _, frame = self.cap.read()

        # Flip frame for sake of mirroring
        frame = cv2.flip(frame, 1)

        # Rescale for efficiency
        frame = cv2.resize(frame, (self.resize_width, self.resize_height), interpolation = cv2.INTER_AREA)

        # Transform colours
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if update_frame:
            self._frame = hsv

        return hsv

    def filter_color(self, frame: np.ndarray, color_boundary: ColorBoundary) -> np.ndarray:
        """
        Filter the last frame for colors that match the boundaries

        :param frame: Frame to check
        :type frame: np.ndarray
        :param color_boundary: The color boundary
        :type color_boundary: ColorBoundary
        :return: np.ndarray
        :rtype: The resultant frame
        """
        # Find colours.
        lower, upper = color_boundary

        # Use a mask to find all that are in the range
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(frame, np.array(lower), np.array(upper))
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        return masked_frame
    
    def find_basic_rect(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Find a basic rect containing all pixels of the color (this function
        takes a filtered frame!)

        This returns None if no colored pixels were found.

        :param frame: The filtered frame to create a rect for
        :type frame: np.ndarray
        :return: The coordinates of the corners of the rect, optional
        :rtype: Tuple[int, int, int, int]
        """
        rows = np.any(frame, axis=0)
        cols = np.any(frame, axis=1)
        try:
            rows_min, rows_max = np.where(rows)[0][[0, -1]]
            cols_min, cols_max = np.where(cols)[0][[0, -1]]
        except IndexError:
            # None was found
            return None
        else:
            return rows_min, cols_min, rows_max, cols_max

    def _acquire_camera(self, camera_id: int = 0):
        """
        Acquire the camera from open-cv2

        :param camera_id: The camera index, defaults to 0
        :type camera_id: int, optional
        """
        # Acquire camera
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    
    def _release_camera(self):
        """
        Release the camera if it is currently acquired.

        If no camera is acquired, this does nothing.
        """
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

    def feed_basic_rect_data(self, color_boundaries: List[ColorBoundary]):
        """
        Feed back coordinates until the loop returns anything that is truthy

        This code is a good example of the appropriate event loop

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
                    color_boundary: None
                    for color_boundary in color_boundaries
                }

                # Get data
                data = self.grab_frame()

                # Filter colors
                for color_boundary in color_boundaries:
                    filtered_frames[color_boundary] = self.filter_color(data, color_boundary)
                
                    # Find basic rects
                    rect_results[color_boundary] = self.find_basic_rect(filtered_frames[color_boundary])
                
                yield filtered_frames, rect_results
        finally:
            self._release_camera()
