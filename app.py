import os
import re
import json
import requests
from datetime import date
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import edge_tts
from fastapi import FastAPI, UploadFile, File

try:
    from ddgs import DDGS
except Exception:
    DDGS = None

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "qwen/qwen3.8-27b"

app = FastAPI()
history = []

SYSTEM_PROMPT = f"""You are Kutty, Viki's friendly personal AI assistant. Today's date is {date.today().strftime('%d %B %Y')}.
- Always reply in simple, natural Tanglish (Tamil written in English letters) unless the user specifically asks for English.
- Keep replies short (1 to 3 sentences), warm, and conversational.
- NEVER use markdown symbols like ** or * or # in your replies.
- The user's name is Viki and his city is Tiruvallur."""

def fetch_weather():
    try:
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 13.1439, "longitude": 79.9089,
                "current": "temperature_2m,relative_humidity_2m,weather_code",
                "daily": "precipitation_probability_max",
                "timezone": "auto", "forecast_days": 1
            }, timeout=5
        ).json()
        codes = {0: "Clear sky", 1: "Mostly clear", 2: "Partly cloudy", 3: "Cloudy",
                 45: "Fog", 51: "Light drizzle", 61: "Light rain", 63: "Rain",
                 65: "Heavy rain", 80: "Showers", 95: "Thunderstorm"}
        c, d = w["current"], w["daily"]
        return {
            "city": "Tiruvallur",
            "temp": round(c["temperature_2m"], 1),
            "desc": codes.get(c["weather_code"], "Clear"),
            "humidity": c["relative_humidity_2m"],
            "rain": d["precipitation_probability_max"][0],
        }
    except Exception:
        pass

    try:
        r = requests.get("https://wttr.in/Tiruvallur?format=j1", timeout=6).json()
        cur = r["current_condition"][0]
        return {
            "city": "Tiruvallur",
            "temp": cur["temp_C"],
            "desc": cur["weatherDesc"][0]["value"],
            "humidity": cur["humidity"],
            "rain": r["weather"][0]["hourly"][0].get("chanceofrain", 0),
        }
    except Exception:
        return {"city": "Tiruvallur", "temp": "--", "desc": "Offline", "humidity": 0, "rain": 0}

def fetch_news():
    if DDGS:
        try:
            with DDGS() as ddgs:
                items = list(ddgs.news("Tamil Nadu news", region="in-en", max_results=3))
            if items:
                return [{"title": i.get("title", "")[:48], "date": i.get("date", "")[:10]} for i in items]
        except Exception:
            pass
    return [{"title": "Tamil Nadu weather & monsoon update", "date": "Today"}]

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard")
def get_dashboard():
    return {"weather": fetch_weather(), "news": fetch_news()}

@app.post("/clear")
def clear_history():
    history.clear()
    return {"status": "cleared"}

@app.post("/speak")
async def speak_audio(req: ChatRequest):
    clean = re.sub(r"[*#_`>|~]", "", req.message).strip()[:350]
    communicate = edge_tts.Communicate(clean, "ta-IN-PallaviNeural")
    async def audio_stream():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    return StreamingResponse(audio_stream(), media_type="audio/mpeg")

@app.post("/chat")
def chat(req: ChatRequest):
    user_msg = req.message.strip()
    history.append({"role": "user", "content": user_msg})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if any(w in user_msg.lower() for w in ("weather", "mazhai", "rain", "veyil", "temp")):
        w = fetch_weather()
        messages.append({
            "role": "system",
            "content": f"LIVE weather for {w['city']}: {w['desc']}, {w['temp']}°C, humidity {w['humidity']}%, rain chance {w['rain']}%. Report in short Tanglish."
        })
    elif DDGS and any(w in user_msg.lower() for w in ("news", "update", "price", "petrol", "gold", "today", "latest")):
        try:
            with DDGS() as ddgs:
                res = list(ddgs.news(user_msg, region="in-en", max_results=3))
            if res:
                summary = " | ".join(f"{r.get('title','')}: {r.get('body','')[:120]}" for r in res)
                messages.append({"role": "system", "content": f"LIVE NEWS: {summary}. Summarize in short Tanglish."})
        except Exception:
            pass

    messages += history[-14:]

    def generate():
        full = ""
        in_think = False
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.3, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if "<think>" in delta:
                in_think = True
                continue
            if "</think>" in delta:
                in_think = False
                continue
            if not in_think and delta:
                full += delta
                yield delta
        history.append({"role": "assistant", "content": full.strip()})

    return StreamingResponse(generate(), media_type="text/plain")
