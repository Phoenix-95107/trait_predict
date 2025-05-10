from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import uvicorn
import json
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import numpy as np
from predict import ocean_predict
from section.relationship_empathy import calculate_section1
from section.work_DNA_focus import calculate_section2
from section.creativity_pulse import calculate_section3
from section.stress_resilience import calculate_section4
import tempfile
from feat import Detector
import concurrent.futures

detector = Detector()
load_dotenv()
app = FastAPI()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


import concurrent.futures

@app.post("/freemium_analyze")
async def detect_trait(files: List[UploadFile] = File(...)):
    # [Validation code remains the same]
    if len(files) < 5:
        raise HTTPException(status_code = 400, detail = "Please input more images. Minimum 5 images required.")
    if len(files) > 7:
        raise HTTPException(status_code = 400, detail = "Please input less images. Maximum 7 images allowed.")
    tmp_files = []
    dataset = []
    
    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare images
        for file in files:
            image_rgb = Image.open(file.file).resize((500,500)).convert('RGB')
            tmp = tempfile.NamedTemporaryFile(delete=False)
            image_rgb.save(tmp.name, format='JPEG')
            tmp_files.append(tmp.name)
            
            image = Image.open(file.file).resize((208, 208)).convert('L')
            data = np.array(image, dtype=np.float32) / 255.0
            data = np.expand_dims(data, axis=0)
            dataset.append(data)
        
        # Detect AU values once for all images
        au_values = detector.detect_image(tmp_files)
        # Run sections in parallel
        futures = [
            executor.submit(calculate_section1, tmp_files, au_values),
            executor.submit(calculate_section2, tmp_files, au_values),
            executor.submit(calculate_section3, tmp_files, au_values),
            executor.submit(calculate_section4, tmp_files, au_values),
        ]
        
        # Collect results
        section_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # # Process OCEAN prediction
    # dataset = np.stack(dataset, axis=0)
    # result = ocean_predict(dataset)
    # output = {
    #     'O': result[0],
    #     'C': result[1],
    #     'E': result[2],
    #     'A': result[3],
    #     'N': result[4]
    # }
    
    return section_results
