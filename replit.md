# Floor Plan Wall Coordinate Extractor

A computer vision tool that extracts wall coordinates from floor plan images using OpenCV.

## Overview

Processes apartment floor plan images to detect wall boundaries and output structured coordinate data.

**Pipeline:**
1. `opencv_solution.py` — reads images from `schemes/`, applies filters (grayscale, thresholding, morphology, Canny edge detection), extracts wall coordinates, and saves them to `coordinates_of_walls.json`
2. `visualization_json.py` — reads the JSON and draws detected walls as red lines on the original images, saving results to `detected_schemes/`

## Project Layout

- `schemes/` — input floor plan images
- `detected_schemes/` — output annotated images
- `coordinates_of_walls.json` — extracted wall coordinate data
- `opencv_solution.py` — main processing script
- `visualization_json.py` — visualization script

## Running

The workflow runs both scripts in sequence:
```
python3 opencv_solution.py && python3 visualization_json.py
```

Or run individually:
```bash
python3 opencv_solution.py      # extract coordinates
python3 visualization_json.py   # generate annotated images
```

## Dependencies

- Python 3.12
- opencv-python 4.13
- numpy 2.4

## User preferences
