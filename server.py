import os
import uuid
import socket
import base64
import datetime
from typing import Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, status, Header, Request
from fastapi.responses import HTMLResponse

app = FastAPI(title="ВКС Телемедицина")

# Инициализируем детектор лиц OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

rooms: Dict[str, Dict[str, Any]] = {}

def load_template(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "templates", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --- Страницы ---

@app.get("/doctor", response_class=HTMLResponse)
def get_doctor_page():
    return load_template("doctor.html")

@app.get("/patient", response_class=HTMLResponse)
def get_patient_page():
    return load_template("patient.html")

# --- API ---

@app.post("/rooms", status_code=status.HTTP_201_CREATED)
def create_room():
    room_id = uuid.uuid4().hex[:6].upper()
    doctor_token = uuid.uuid4().hex
    rooms[room_id] = {
        "doctor_token": doctor_token,
        "patient_token": None,
        "doctor_events": [],
        "patient_events": []
    }
    return {"room": room_id, "token": doctor_token}

@app.post("/rooms/{room}/join", status_code=status.HTTP_200_OK)
def join_room(room: str):
    if room not in rooms:
        raise HTTPException(status_code=404, detail="Комната не найдена")

    room_data = rooms[room]
    if room_data["patient_token"] is not None:
        raise HTTPException(status_code=400, detail="Комната уже занята")

    patient_token = uuid.uuid4().hex
    room_data["patient_token"] = patient_token
    return {"token": patient_token}

def authenticate(room: str, authorization: str) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Отсутствует Bearer токен")
    token = authorization.split("Bearer ")[1].strip()
    if room not in rooms:
        raise HTTPException(status_code=404, detail="Комната не найдена")

    if token == rooms[room]["doctor_token"]:
        return "doctor"
    elif token == rooms[room]["patient_token"]:
        return "patient"
    raise HTTPException(status_code=403, detail="Неверный токен")

@app.post("/rooms/{room}/events", status_code=status.HTTP_202_ACCEPTED)
def send_event(room: str, event: dict, authorization: str = Header(None)):
    role = authenticate(room, authorization)
    target_queue = "patient_events" if role == "doctor" else "doctor_events"
    rooms[room][target_queue].append(event)
    return {"status": "queued"}

@app.get("/rooms/{room}/events", status_code=status.HTTP_200_OK)
def get_events(room: str, after: int = 0, authorization: str = Header(None)):
    role = authenticate(room, authorization)
    queue = rooms[room]["doctor_events"] if role == "doctor" else rooms[room]["patient_events"]
    return {
        "events": queue[after:],
        "next_index": len(queue)
    }

# --- Модуль анализа видеокадров ---

@app.post("/rooms/{room}/frames", status_code=status.HTTP_200_OK)
async def analyze_frame(room: str, request: Request, authorization: str = Header(None)):
    """
    Принимает JPEG от пациента, находит лица, обводит рамкой
    и передает обработанный кадр врачу.
    """
    role = authenticate(room, authorization)
    if role != "patient":
        raise HTTPException(status_code=403, detail="Только пациент отправляет кадры")

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Пустой кадр")

    # Преобразуем байты в изображение OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Ошибка декодирования JPEG")

    # Поиск лиц (в градациях серого)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # Рисуем зеленые рамки вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Patient Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Кодируем обратно в JPEG для передачи врачу
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    b64_image = base64.b64encode(buffer).decode('utf-8')

    result = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "faces_count": len(faces),
        "face_detected": len(faces) > 0,
        "image": f"data:image/jpeg;base64,{b64_image}"
    }

    # Кладем результат в события для врача
    rooms[room]["doctor_events"].append({
        "type": "face_analysis",
        "data": result
    })

    return {"status": "ok", "faces_count": len(faces)}

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    port = 8000

    ssl_kwargs = {}
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        ssl_kwargs = {"ssl_certfile": "cert.pem", "ssl_keyfile": "key.pem"}

    print("\n" + "="*55)
    print("🚀 СЕРВЕР ЗАПУЩЕН")
    print(f"💻 На ПК (Врач):          https://localhost:{port}/doctor")
    print(f"📱 На Телефоне (Пациент): https://{local_ip}:{port}/patient")
    print("="*55 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, **ssl_kwargs)