from fastapi import FastAPI
from fastapi import UploadFile, File
from .model import Generator
import io
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import base64
import torch
import torchvision
import pickle
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

def from_image_to_bytes(img):
    """
    pillow image to bytes
    """
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()

    encoded = base64.b64encode(imgByteArr)
    decoded = encoded.decode('ascii')
    return decoded


@app.get("/generate/")
def generate():
    G = Generator(image_size=512, attn_res_layers=[32]) #pickle.load(open('model.pkl', 'rb'))
    state_dict = pickle.load(open('./app/model.pkl', 'rb'))
    G.load_state_dict(state_dict, strict=True)
    latents = torch.randn((1, 256))
    if torch.cuda.is_available():
        G.cuda()
        latents = latents.cuda()
    G.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            generated_image = G(latents).clamp_(0., 1.).cpu()

    image = torchvision.transforms.ToPILImage()(generated_image.squeeze()) #generated_image.squeeze().numpy()
    
    image_array = np.array(image)
    image_array = cv2.GaussianBlur(image_array,(5,5),cv2.BORDER_DEFAULT)#cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)
    
    return [from_image_to_bytes(Image.fromarray(image_array))]
