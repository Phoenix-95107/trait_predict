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
import matplotlib.pyplot as plt
from feat import Detector
import torch
import tempfile

detector = Detector()

load_dotenv()
app = FastAPI()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.post("/freemium_analyze")
async def detect_trait(files: List[UploadFile] = File(...)):
    num_files = len(files)
    if num_files < 5 or num_files > 7:
        raise HTTPException(
            status_code=400,
            detail=
            f"Number of imagese be between 5 and 7. You uploaded {num_files}.")
    tmp_files = []
    dataset = []
    for file in files:
        #
        image_rgb = Image.open(file.file).resize((500,500)).convert('RGB')
        tmp = tempfile.NamedTemporaryFile(delete=False)
        image_rgb.save(tmp.name, format='JPEG')
        tmp_files.append(tmp.name)
        image = Image.open(file.file).resize((208, 208)).convert('L')
        data = np.array(image, dtype=np.float32) / 255.0
        data = np.expand_dims(data, axis=0)
        dataset.append(data)
        
    au_result = detector.detect_image(tmp_files)
    au_result.to_csv("au_result.csv")
    au_value = np.mean(au_result.aus, axis=0)
    print(f"au_result is {au_value}")
    dataset = np.stack(dataset, axis=0)
    result = ocean_predict(dataset)
    output = {
        'O': result[0],
        'C': result[1],
        'E': result[2],
        'A': result[3],
        'N': result[4]
    }
    response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.0,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a behavioral personality analysis engine trained in trait theory, including OCEAN (Big Five) and AU (Action Unit). "
                "Given normalized OCEAN and AU scores (range 0–1), you will return a structured psychological profile. "
                "Each section must be concise, interpretable, and directly derived from trait interactions. Respond strictly in valid JSON format."
            )
        },
        {
            "role": "user",
            "content": f"""
      Input:OCEAN values: {output}
      AU values (normalized):{au_value}

      Output Format: JSON
      Generate the following attributes:
      1. Top Traits(A Sneak Peek):(according AU and OCEAN values)
        - Show highest and lowest values(e.g. Creativity:high(85/100), Empathy:Moderate(68/100))
        - Key Wvidence Highlights
          - Signs of genuine positive emotion detected.
          - Facial cues suggesting openness and approachability.
          - Subtle non-verbal signals of engagement observed.
        - Why It Matters(Just the Start)
            e.g. Your facial cues suggest a natural openness and curiosity, often seen in individuals who enjoy exploring new ideas and connecting with others. This is just the surface – your premium report reveals the full depth and nuances.
        - Benchmark(Partial View)
          e.g. Your **Creativity** score is higher than 75% of users in your age group.

      2. Personality Snapshot(Strengths & Growth Areas)
        
      3. Unique Personality Story:
          e.g. Based on the intricate patterns and subtle cues captured from your photos, your personality tells a story of vibrant curiosity and genuine connection. Your expressions reveal a natural inclination towards empathy, allowing you to easily understand and resonate with the emotions of those around you. This is beautifully complemented by an adventurous spirit, reflected in the dynamic aspects of your facial cues, suggesting a readiness to explore new horizons and embrace novel experiences. Your profile paints a picture of someone who navigates the world with both a warm heart and an open mind, constantly seeking to learn and connect on a deeper level. This blend of emotional intelligence and exploratory drive makes you a truly unique individual, capable of building meaningful relationships and finding inspiration in the unexpected.
          
      4. Relationships & Empathy
        - Eyes - Iris-to-sclera ratio: +1.4 SD (This measures how much of the white part of your eye is visible compared to the colored part, suggesting an open and approachable look.)
        - AU 6 (Cheek Raiser) + AU 12 (Lip Corner Puller): Frequent co-occurrence (This indicates the presence of genuine, "Duchenne" smiles, detected across 4 of your photos.)
        - AU 1 (Inner Brow Raiser): Short duration lifts (Subtle, brief lifts of your inner eyebrows were detected, which can show attention or sympathy.)
        - Gaze: Direct and consistent (You maintained direct eye contact in 5 out of your 6 photos.)
        - Show Score Breakdown:
          - Trust Signaling: 0-100(high,moderate,moderate-low)
          - Social Opennesss: 0-100(high,moderate,moderate-low)
          - Empathy index: 0-100(high,moderate,moderate-low)
          - Conflict Avoidance: 0-100(high,moderate,moderate-low)
        - Actionable Steps for Development:like this
            **To Leverage Trust Signaling & Social Openness:** Actively participate in social events or online communities where genuine connection is valued. Practice initiating conversations with new people.
            **To Develop Empathy:** Practice active listening, focusing on understanding others' perspectives without interruption. Read fiction to step into different characters' shoes.
            **To Improve Conflict Navigation:** Learn basic assertive communication techniques (e.g., using "I feel" statements). Practice setting small boundaries in low-stress situations. Role-play difficult conversations with a trusted friend.
        - Your **Empathy Index** is 10% higher than the average for your age group.
      5. Work DNA & Focus
        - influence_score: integer (0–100), based on dominance, agency, and motivation cues.

      6. Collaboration & Friendship:
        - social_energy: qualitative descriptor (e.g. "Energized in groups", "Selective engager")
        - loyalty_signals: brief description
        - conflict_resolution_style: short label or phrase

      7. Relationships & Empathy:
        - trust_signals: 0-100(high, moderate, moderate-low)
        - Social Openness: 0-100(high, moderate, moderate-low)
        - Empathy index: 0-100(high, moderate, moderate-low)
        - Conflict Avoidance:0-100(high, moderate, moderate-low)

      8. Adventure & Risk:
        - novelty_seeking: descriptor
        - calculated_risk_index: brief explanation
        - spontaneity_score: low/medium/high
        - risk_meter: 0–100 numeric scale

      9. Creativity Pulse:
        - divergent_thinking_index: short label or explanation
        - curiosity_spark: qualitative descriptor
        - aesthetic_sensitivity: brief description
        - overall_creativity_score: 0–100

      10. Stress & Resilience:
        - tension_baseline: short label (e.g. "Low", "Moderate", "High")
        - recovery_speed: brief note
        - resilience_label: 1-word label (e.g. "Resilient Moderator")
        - resilience_level: 0–100

      11. Learning & Growth:
        - preferred_learning_style: "Visual" | "Social" | "Experiential" | etc.
        - feedback_receptiveness: 1-sentence descriptor

      12. Compatibility Snapshot (Optional):
        - best_fit_types: MBTI or Enneagram types (string)
        - compatibility_score: 0–100

      13. Historical Trend:
        If user retakes: spark-line of trait changes over time.
          - ocean_sparkline: [simulated array of OCEAN values over time]
          - trend_note: short sentence (e.g. "Openness rising steadily")

      14. Featured Quote:
        - quote: 25-word motivational quote written in second person, inspired by traits.

      15. Confidence & Limits:
        - disclaimer: short note on trait model limitations
        - accuracy_note: sentence on recommended self-assessment

      Instructions:
      - Use neutral-professional tone.
      - Follow valid JSON syntax.
      - Reflect logic from both OCEAN and AU traits accurately.
      - Avoid generic filler content.
      - Ensure output is cleanly structured for UI rendering.
      """
              }
          ]
      )

    trait = response.choices[0].message.content.replace("```","").replace("json","")
    trait = json.loads(trait)
    print(trait)
    return trait


@app.post("/freemium_analyze_test")
async def detect_trait(file: UploadFile = File(...)):  
    image = Image.open(file.file).resize((500, 500)).convert('L')
    # plt.imshow(image)
    # plt.show()
    data = np.array(image, dtype=np.float32) / 255.0
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=0)
    print(f'image size is {data.shape}')
    result = ocean_predict(data)
    return {
        'O_pred': result[0],
        'C_pred': result[1],
        'E_pred': result[2],
        'A_pred': result[3],
        'N_pred': result[4]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)