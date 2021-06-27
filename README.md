# Color detection program
This is a set of scripts to detect colors from the camera and then turn them into virtual coordinates. This uses:
- `opencv2` (camera interface)
- `numpy` (image data management)
- `scikit` (distinguishing color clusters)
- `matplotlib` (tests) 
- `pygame` (also tests)

## Installation
Install the requirements with
```
$ python3 -m pip install -r requirements.txt
```

Or the following on Windows

```
$ py -m pip install -r requirements.txt
```

## Usage
You can import these libraries, but for example usage, see [Examples](#examples).

## Examples
Run all of these examples with `python -m examples.<name>`.

### `cluster_detection`
Uses `ClusterDetection.feed_cluster_rect_data` to find all clusters of the color and then draws rects around each cluster found.

### `key_point_detection`
Uses `ClusterDetection.find_key_points` to find the largest areas of the wanted colors and then draws lines between them. This is the matplotlib simplified version of `pygame_test`.

### `pygame_test`
Tests usage of normalized points with Pygame, and draws circles and rects around red and blue points and recognizes human skin.

This also draws a line between key points.
