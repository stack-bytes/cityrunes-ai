from fastapi import FastAPI, HTTPException, UploadFile

from app.comparison import ComparisonResult, ImageDecodeError, compare_images

app = FastAPI(title="CityRunes Image Comparison")


@app.post("/compare", response_model=ComparisonResult)
def compare(image_a: UploadFile, image_b: UploadFile) -> ComparisonResult:
    try:
        return compare_images(image_a.file.read(), image_b.file.read())
    except ImageDecodeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
