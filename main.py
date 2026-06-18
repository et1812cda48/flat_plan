import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Floor Plan Wall Extractor",
    description="Upload a floor plan image and receive the coordinates of detected walls.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _process_image_bytes(data: bytes) -> list:
    arr = np.frombuffer(data, np.uint8)
    scheme = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if scheme is None:
        raise ValueError("Could not decode image. Make sure you uploaded a valid PNG or JPEG.")

    scheme = cv2.cvtColor(scheme, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(scheme, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    walls = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    coords = []
    edges = cv2.Canny(walls, 10, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    delta = 10
    for contour in contours:
        points = contour[:, 0, :]
        n_points = len(points)
        if n_points < 2:
            continue
        for i in range(n_points - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            dist = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
            if dist >= delta:
                coords.append([[int(x0), int(y0)], [int(x1), int(y1)]])
        if n_points > 2:
            x0, y0 = points[-1]
            x1, y1 = points[0]
            dist = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
            if dist >= delta:
                coords.append([[int(x0), int(y0)], [int(x1), int(y1)]])
    return coords


@app.get("/", summary="Health check")
def health_check():
    return {"status": "ok"}


@app.post(
    "/api/v1/process",
    summary="Process a floor plan image",
    description=(
        "Upload a floor plan image (PNG, JPG, BMP, TIFF) and receive a list of "
        "detected wall segments as pairs of [x, y] coordinate points."
    ),
)
async def process_floor_plan(file: UploadFile = File(...)):
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload a PNG or JPEG.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        coords = _process_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    walls = [{"id": f"w{i + 1}", "points": pts} for i, pts in enumerate(coords)]
    return JSONResponse(
        content={
            "filename": file.filename,
            "wall_count": len(walls),
            "walls": walls,
        }
    )
