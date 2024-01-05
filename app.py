from main import process_image, downloadImage
import uuid
from ResponseBody import BodyAnalysisResult
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
filename = f"body-{uuid.uuid4()}.jpg"


@app.get("/")
def root():
    return {"message": "Welcome to Body Analysis Tool"}


@app.post("/check_image_download")
def check_image_download(url: str):
    resultant_boolean = downloadImage(url, filename)
    return {"imageDownloaded": str(resultant_boolean)}


@app.get("/get_client_ip")
def get_client_ip(request: Request):
    client_ip = request.client.host
    return {"client_ip": client_ip}


@app.post("/result", response_model=BodyAnalysisResult)
def result():
    res_waist_coordinate, res_waist_message, res_shoulder_coordinate, res_shoulder_message = process_image(filename)
    return dict(messageOnShoulder=res_shoulder_message, shoulderCoordinate=res_shoulder_coordinate,
                messageOnWaist=res_waist_message, waistCoordinate=res_waist_coordinate)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
