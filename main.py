from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import gdown
import os
import zipfile
import uuid
import random
from fastapi import Request
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from database import init_db
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# =========================
# App Setup
# =========================

app = FastAPI()
init_db()
os.makedirs("sessions", exist_ok=True)
app.mount("/sessions", StaticFiles(directory="sessions"), name="sessions")

templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload-page", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/sessions-page", response_class=HTMLResponse)
async def sessions_page(request: Request):
    sessions = os.listdir("sessions")
    return templates.TemplateResponse(
        "sessions.html",
        {"request": request, "sessions": sessions}
    )


@app.get("/results-page", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "results": []}
    )

# =========================
# Model Versioning Setup
# =========================

NUM_CLASSES = 10
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)


def get_next_version():
    existing = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("model_v") and f.endswith(".pth")
    ]

    if not existing:
        return 1

    nums = [int(f.split("_v")[1].split(".pth")[0]) for f in existing]
    return max(nums) + 1


def get_latest_model_path():
    latest_file = os.path.join(MODELS_DIR, "latest.txt")

    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            name = f.read().strip()
            path = os.path.join(MODELS_DIR, name)
            if os.path.exists(path):
                return path

    # fallback لو مفيش latest.txt
    existing = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("model_v") and f.endswith(".pth")
    ]

    if not existing:
        return None

    nums = sorted(
        [int(f.split("_v")[1].split(".pth")[0]) for f in existing]
    )

    return os.path.join(MODELS_DIR, f"model_v{nums[-1]}.pth")


# =========================
# تحميل أحدث موديل
# =========================

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model_path = "models/smartlabeler_model.pth"

if not os.path.exists(model_path):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1uVrMGispCULBL7LuULxW0RNpUr-2"
    gdown.download(url, model_path, quiet=False)
latest_model = get_latest_model_path()

if latest_model:
    print(f"Loading model: {latest_model}")
    state = torch.load(latest_model, map_location="cpu")
    model.load_state_dict(state)
else:
    print("No previous model found. Starting fresh.")

model.eval()


# =========================
# Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


# =========================
# Dataset من labels.txt
# =========================

class LabeledDataset(Dataset):
    def __init__(self, labels_file, transform=None):
        self.samples = []
        self.transform = transform

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.split(",")
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# الصفحة الرئيسية
# =========================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# =========================
# رفع الداتا
# =========================

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):

    session_id = str(uuid.uuid4())
    session_path = os.path.join("sessions", session_id)
    os.makedirs(session_path, exist_ok=True)

    zip_path = os.path.join(session_path, file.filename)

    with open(zip_path, "wb") as f:
        content = await file.read()
        f.write(content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(session_path)

    return {"session_id": session_id}


# =========================
# Active Learning Run
# =========================

@app.get("/run/{session_id}", response_class=HTMLResponse)
async def run_model(request: Request, session_id: str):

    session_path = os.path.join("sessions", session_id)

    if not os.path.exists(session_path):
        return HTMLResponse("Session not found")

    labeled_set = set()
    labels_file = os.path.join(session_path, "labels.txt")

    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                path, _ = line.strip().split(",")
                labeled_set.add(path)

    all_images = []

    for root, dirs, files in os.walk(session_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file).replace("\\", "/")
                if img_path not in labeled_set:
                    all_images.append(img_path)

    if len(all_images) == 0:
        return HTMLResponse("<h3>All images labeled ✔</h3>")

    random.shuffle(all_images)
    sample_images = all_images[:300]

    results = []

    for img_path in sample_images:
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        results.append({
            "image": img_path,
            "predicted_class": int(predicted.item()),
            "confidence": round(float(confidence.item()), 3)
        })

    results.sort(key=lambda x: x["confidence"])

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "results": results[:5],
            "session_id": session_id
        }
    )


# =========================
# حفظ اللابلز
# =========================

@app.post("/save_labels/{session_id}", response_class=HTMLResponse)
async def save_labels(request: Request, session_id: str):

    form = await request.form()

    session_path = os.path.join("sessions", session_id)
    labels_file = os.path.join(session_path, "labels.txt")

    with open(labels_file, "a", encoding="utf-8") as f:
        for key in form.keys():
            if key.startswith("label_"):
                index = key.split("_")[1]
                label = form[key]
                image_path = form.get(f"image_{index}")

                if image_path:
                    f.write(f"{image_path},{label}\n")

    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Labels Saved</title>

<style>
body {{
    margin:0;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto;
    background:#0f172a;
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
    color:white;
}}

.card {{
    background:#1e293b;
    padding:40px;
    border-radius:20px;
    width:420px;
    text-align:center;
    box-shadow:0 25px 60px rgba(0,0,0,.4);
}}

.icon {{
    font-size:60px;
    margin-bottom:20px;
}}

h2 {{
    margin:0;
}}

p {{
    color:#94a3b8;
    font-size:14px;
    margin-top:10px;
}}

.buttons {{
    margin-top:30px;
    display:flex;
    gap:15px;
    justify-content:center;
}}

button {{
    padding:10px 20px;
    border:none;
    border-radius:10px;
    font-weight:600;
    cursor:pointer;
}}

.primary {{
    background:linear-gradient(90deg,#6366f1,#22d3ee);
    color:white;
}}

.secondary {{
    background:#334155;
    color:white;
}}

.loading {{
    display:none;
    margin-top:15px;
    font-size:13px;
    color:#94a3b8;
}}

</style>
</head>

<body>

<div class="card">

<div class="icon">✅</div>

<h2>Labels Saved Successfully</h2>
<p>Your annotations have been stored for this session.</p>

<div class="buttons">
    <button class="primary" onclick="retrain()">Retrain Model</button>
    <button class="secondary" onclick="goBack()">Continue Labeling</button>
</div>

<div class="loading" id="loading">
Training model... please wait ⏳
</div>

</div>

<script>
function goBack(){{
    window.location.href = "/run/{session_id}";
}}

function retrain(){{
    document.getElementById("loading").style.display="block";
    fetch("/retrain/{session_id}", {{method:"POST"}})
    .then(()=>window.location.href="/run/{session_id}");
}}
</script>

</body>
</html>
""")

# =========================
# Retraining (Versioned)
# =========================

@app.post("/retrain/{session_id}")
async def retrain_model(session_id: str):

    session_path = os.path.join("sessions", session_id)
    labels_file = os.path.join(session_path, "labels.txt")

    if not os.path.exists(labels_file):
        return {"error": "No labels found"}

    dataset = LabeledDataset(labels_file, transform=transform)

    if len(dataset) < 5:
        return {"error": "Label more images first"}

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Train FC only
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(3):
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()

    # حفظ نسخة جديدة
    version = get_next_version()
    model_name = f"model_v{version}.pth"
    save_path = os.path.join(MODELS_DIR, model_name)

    torch.save(model.state_dict(), save_path)

    # تحديث latest
    with open(os.path.join(MODELS_DIR, "latest.txt"), "w") as f:
        f.write(model_name)

    print(f"Saved new model version: {model_name}")

    return RedirectResponse(url=f"/run/{session_id}", status_code=303)