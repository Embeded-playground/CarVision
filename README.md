# Parking Vision Service

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-9CA3AF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Layanan computer vision untuk memantau ketersediaan slot parkir. Service ini menggabungkan deteksi kendaraan **Ultralytics YOLO** dengan evaluasi polygon ROI, *temporal smoothing* anti-flicker, REST API FastAPI, autentikasi API key, dan outbound webhook untuk integrasi dengan backend Go maupun web dashboard.

Dirancang untuk CCTV parkir dengan target GPU NVIDIA RTX 4GB (RTX 3050/3060 Laptop) menggunakan bobot `yolo26m.pt`, dengan fallback CPU otomatis.

---

## Fitur Utama

- **Ground contact point `(x_center, y_max)`** — occupancy dinilai dari titik kontak roda/aspal kendaraan, bukan centroid bbox yang rawan melenceng karena perspektif kamera.
- **Fallback IoA vectorized** — jika ground point tidak berada di dalam slot, service memakai Intersection over Area antara footprint kendaraan dan polygon slot.
- **Temporal state buffer** — status slot baru berubah setelah konsisten selama N frame berturut-turut, mencegah flicker akibat okulasi/pedestrian.
- **Confidence per slot** — confidence diambil dari deteksi kendaraan paling kuat yang match dengan slot.
- **CUDA acceleration** — inferensi di `torch.no_grad()` dengan `device='cuda'` jika tersedia; fallback CPU dapat dikonfigurasi.
- **REST API** — health check, status slot, deteksi gambar, dan CRUD konfigurasi slot.
- **API key authentication** — proteksi endpoint `/api/v1/*` melalui header `X-API-Key`.
- **Outbound webhook** — notifikasi asynchronous setiap kali status slot berubah.
- **Normalisasi perspektif opsional** — warp top-down sebelum evaluasi occupancy untuk kamera miring.
- **Editor slot interaktif** — auto-grid, ROI perspektif, dan input polygon manual.

---

## Arsitektur Integrasi Web UI

Pola yang direkomendasikan:

```text
Web UI
  │  polling JSON tiap 1-2 detik (lewat Go backend)
  ▼
Go Backend
  │  POST /api/v1/detect        (kirim frame CCTV + X-API-Key)
  │  GET  /api/v1/slots/status  (ambil snapshot status)
  │  GET  /api/v1/slots         (ambil polygon sekali di awal)
  │  ◄── Webhook slot_state_changed dari service Python
  ▼
FastAPI + YOLO
```

Catatan:

- Web UI tidak memanggil FastAPI secara langsung. `API_KEY` disimpan di Go backend agar tidak terekspos ke browser.
- Karena Web UI dan backend berada di domain/reverse proxy yang sama, CORS tidak diperlukan.
- Go backend bertugas mengirim frame CCTV/RTSP ke `POST /api/v1/detect`.
- Web UI polling status ke Go backend; Go dapat meng-cache hasil dari service Python.
- Jika Web UI ingin menampilkan polygon slot, ambil `GET /api/v1/slots` sekali, lalu polling `GET /api/v1/slots/status` untuk perubahan status.
- Gunakan `/openapi.json` untuk generate client Go.

> **Penting:** `GET /api/v1/slots/status` hanya membaca cache hasil inferensi terakhir. Status akan terus diperbarui selama ada pihak yang mengirim frame ke `/api/v1/detect`.

---

## Struktur Proyek

```text
parking-vision-service/
├── app/
│   ├── __init__.py
│   ├── config.py          # Load .env + konfigurasi bertipe
│   ├── detector.py        # YOLO, ground point, IoA, temporal buffer, CUDA
│   ├── geometry.py        # Polygon vectorized: point-in-poly, IoU/IoA, perspektif
│   ├── main.py            # FastAPI endpoints, API key auth, lifespan
│   ├── schemas.py         # Model Pydantic API & webhook
│   ├── slot_store.py      # Baca/tulis parking_slots.json
│   ├── sources.py         # Auto-discover video test
│   ├── state.py           # Cache snapshot status slot (thread-safe)
│   └── webhook.py         # Async webhook queue + retry
├── models/                # Bobot YOLO (.pt)
├── test_videos/           # Video test (gitignored)
├── tests/                 # Unit test geometry, temporal, API, webhook
├── slot_editor.py         # GUI editor slot interaktif
├── test_video.py          # Demo deteksi real-time
├── parking_slots.json     # Konfigurasi polygon ROI
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Alur Deteksi

```text
Frame
  → YOLO inference (imgsz=1280, confidence per-kelas)
  → Filter kelas kendaraan (car/motorcycle/bus/truck)
  → Ground contact point (x_center, y_max) di dalam polygon?
      ya  → slot terisi
      tidak → IoA bbox vs polygon >= OCCUPANCY_IOA_THRESHOLD?
                ya  → slot terisi
                tidak → slot kosong
  → Temporal buffer N frame
  → Confidence per slot
  → Response API + update cache + webhook (jika status berubah)
```

---

## Prasyarat dan Setup

### Kebutuhan Sistem

- Python 3.10+
- NVIDIA GPU (disarankan) dengan driver CUDA compatible
- Windows / Linux

### 1. Clone dan Virtual Environment

```bash
git clone <repo-url> parking-vision-service
cd parking-vision-service
python -m venv venv
```

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Bash:

```bash
source venv/bin/activate
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

Untuk menjalankan test:

```bash
pip install -r requirements-dev.txt
```

> Fitur GUI (`slot_editor.py`, `test_video.py`) membutuhkan OpenCV dengan backend GUI.
> Jika memakai `opencv-python-headless`:
> ```bash
> pip uninstall -y opencv-python-headless
> pip install opencv-python
> ```

### 3. PyTorch + CUDA

Build PyTorch default dari PyPI adalah CPU. Untuk GPU NVIDIA:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verifikasi:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4. Konfigurasi `.env`

```bash
cp .env.example .env
```

Sesuaikan minimal:

```env
MODEL_PATH=models/yolo26m.pt
DEVICE=cuda
ALLOW_CPU_FALLBACK=true
API_KEY=change-me
WEBHOOK_URL=http://localhost:8080/webhook/parking
```

### 5. Bobot Model

Letakkan bobot YOLO di folder `models/`:

```text
models/
├── yolo26m.pt   # default, direkomendasikan untuk 4GB VRAM
├── yolo26n.pt   # fallback ringan
├── yolo26x.pt   # akurasi tinggi, butuh VRAM besar
└── yolov8n.pt
```

Ultralytics dapat mengunduh bobot otomatis jika file belum tersedia.

---

## Konfigurasi Environment

| Variabel | Fungsi | Default |
|---|---|---|
| `MODEL_PATH` | Path bobot YOLO | `models/yolo26m.pt` |
| `SLOTS_PATH` | Path konfigurasi slot | `parking_slots.json` |
| `DEVICE` | Device inferensi: `cuda`, `cuda:0`, atau `cpu` | `cuda` |
| `ALLOW_CPU_FALLBACK` | Fallback ke CPU jika CUDA tidak tersedia | `true` |
| `YOLO_IMGSZ` | Ukuran letterbox inference | `1280` |
| `OCCUPANCY_IOA_THRESHOLD` | Ambang IoA kendaraan vs slot | `0.30` |
| `SLOT_STATE_FRAMES` | Jumlah frame konsisten sebelum status berubah | `5` |
| `SLOT_OCCUPIED_FRAMES` | Override konfirmasi transisi terisi | `SLOT_STATE_FRAMES` |
| `SLOT_EMPTY_FRAMES` | Override konfirmasi transisi kosong | `SLOT_STATE_FRAMES` |
| `PARKING_PERSPECTIVE` | 4 titik warp top-down: `x1,y1,...,x4,y4` | kosong (nonaktif) |
| `API_KEY` | API key untuk endpoint `/api/v1/*` | kosong (auth nonaktif) |
| `CORS_ORIGINS` | Daftar origin CORS (pisahkan koma, atau `*`) | `*` |
| `WEBHOOK_URL` | URL tujuan outbound webhook | kosong (nonaktif) |
| `WEBHOOK_API_KEY` | API key untuk webhook; fallback ke `API_KEY` | `API_KEY` |
| `WEBHOOK_TIMEOUT` | Timeout HTTP webhook (detik) | `5` |
| `WEBHOOK_RETRIES` | Jumlah retry webhook | `3` |
| `TEST_VIDEO_DIR` | Folder video test | `test_videos` |
| `DEFAULT_VIDEO` | Video default saat prompt | video pertama ditemukan |


Contoh `PARKING_PERSPECTIVE`:

```env
PARKING_PERSPECTIVE=100,900,1820,900,1500,300,420,300
```

---

## Menjalankan Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Liveness: `http://127.0.0.1:8000/`

---

## Autentikasi API

Jika `API_KEY` di-set, semua endpoint `/api/v1/*` membutuhkan header:

```http
X-API-Key: your-secret-key
```

Jika `API_KEY` kosong, autentikasi dinonaktifkan (khusus development).

Endpoint `GET /` tetap terbuka untuk liveness/load balancer.

---

## API Endpoints

| Method | Endpoint | Auth | Deskripsi |
|---|---|---|---|
| `GET` | `/` | Tidak | Liveness sederhana untuk load balancer |
| `GET` | `/api/v1/health` | Ya | Health, model, CUDA, auth, webhook, dan last_inference |
| `GET` | `/api/v1/slots/status` | Ya | Snapshot status slot real-time + last_inference |
| `POST` | `/api/v1/detect` | Ya | Deteksi dari upload gambar/frame CCTV |
| `GET` | `/api/v1/slots` | Ya | Ambil seluruh konfigurasi polygon slot |
| `GET` | `/api/v1/slots/{slot_id}` | Ya | Ambil konfigurasi satu slot |
| `POST` | `/api/v1/slots` | Ya | Tambah satu slot baru |
| `PUT` | `/api/v1/slots/{slot_id}` | Ya | Perbarui polygon satu slot |
| `DELETE` | `/api/v1/slots/{slot_id}` | Ya | Hapus satu slot |
| `PUT` | `/api/v1/slots` | Ya | Overwrite seluruh konfigurasi slot (bulk) |
| `DELETE` | `/api/v1/slots` | Ya | Hapus semua slot |

### `GET /api/v1/health`

Contoh:

```bash
curl -H "X-API-Key: $API_KEY" \
  http://127.0.0.1:8000/api/v1/health
```

Response (`200 OK`):

```json
{
  "status": "ok",
  "timestamp": "2026-10-26T17:27:00Z",
  "service": "parking-vision-service",
  "model_loaded": "models/yolo26m.pt",
  "device": "cuda",
  "cuda": {
    "available": true,
    "device_count": 1,
    "device_name": "NVIDIA GeForce RTX 3050 ...",
    "memory_allocated_mb": 123.4,
    "memory_reserved_mb": 234.5
  },
  "auth_enabled": true,
  "webhook_configured": true,
  "last_inference": "2026-10-26T17:26:58Z"
}
```

`status` bernilai `ok` jika inferensi memakai CUDA, dan `degraded` jika memakai CPU.

### `GET /api/v1/slots/status`

Mengembalikan snapshot terakhir status occupancy seluruh slot beserta `last_inference`. Sebelum ada frame yang dikirim ke `/detect`, semua slot bernilai `EMPTY` dengan confidence `0.0` dan `last_inference: null`.

```bash
curl -H "X-API-Key: $API_KEY" \
  http://127.0.0.1:8000/api/v1/slots/status
```

Response (`200 OK`):

```json
{
  "timestamp": "2026-10-26T17:27:00Z",
  "summary": {
    "total_slots": 16,
    "occupied_slots": 4,
    "empty_slots": 12
  },
  "slots": [
    {
      "slot_id": "Slot_1",
      "is_occupied": true,
      "confidence": 0.92
    }
  ],
  "last_inference": "2026-10-26T17:26:58Z"
}
```

> **Deteksi Kamera Offline/Stale oleh Web UI:** Jika `last_inference` bernilai `null` atau selisih dengan waktu sekarang lebih dari 10 detik, Web UI dapat menampilkan status peringatan *"Kamera Offline / Feed Terhenti"*.

### `POST /api/v1/detect`

Query parameter:

| Parameter | Default | Deskripsi |
|---|---|---|
| `annotated` | `false` | Sertakan gambar anotasi base64 |
| `use_temporal` | `true` | Gunakan temporal smoothing. Set `false` untuk one-shot detection |

Contoh:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@frame.jpg"
```

Dengan gambar anotasi:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect?annotated=true" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@frame.jpg"
```

Response (`200 OK`):

```json
{
  "timestamp": "2026-10-26T17:27:00Z",
  "summary": {
    "total_slots": 16,
    "occupied_slots": 4,
    "empty_slots": 12
  },
  "slots": [
    {
      "slot_id": "Slot_1",
      "is_occupied": true,
      "confidence": 0.92
    }
  ],
  "image_annotated": "data:image/jpeg;base64,..."
}
```

### CRUD Slot (Konfigurasi ROI)

Service menyediakan API lengkap untuk manajemen slot, baik secara granular per-slot maupun secara bulk.

Setiap polygon divalidasi ketat:
- Harus terdiri dari tepat 4 titik koordinat `[x, y]`.
- Harus konveks dan tidak boleh saling bersilangan (*self-intersecting / bowtie*).
- Luas area harus berada dalam rentang valid (default `50 - 10.000.000 px²`).
- Jika tidak valid, API mengembalikan `422 Unprocessable Entity` dengan rincian kesalahan spesifik.

#### 1. Ambil Semua Slot (`GET /api/v1/slots`)

```bash
curl -H "X-API-Key: $API_KEY" \
  http://127.0.0.1:8000/api/v1/slots
```

Response (`200 OK`):

```json
{
  "success": true,
  "total": 2,
  "slots": [
    {
      "slot_id": "Slot_1",
      "polygon": [[1336, 333], [1448, 332], [1568, 482], [1428, 486]]
    },
    {
      "slot_id": "Slot_2",
      "polygon": [[940, 328], [1034, 452], [933, 483], [822, 344]]
    }
  ]
}
```

#### 2. Ambil Satu Slot (`GET /api/v1/slots/{slot_id}`)

```bash
curl -H "X-API-Key: $API_KEY" \
  http://127.0.0.1:8000/api/v1/slots/Slot_1
```

Response (`200 OK`):

```json
{
  "slot_id": "Slot_1",
  "polygon": [[1336, 333], [1448, 332], [1568, 482], [1428, 486]]
}
```

Jika tidak ada, mengembalikan `404 Not Found`.

#### 3. Tambah Satu Slot (`POST /api/v1/slots`)

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/slots" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "slot_id": "Slot_3",
    "polygon": [[34, 657], [158, 657], [238, 843], [79, 846]]
  }'
```

Response (`201 Created`):

```json
{
  "success": true,
  "message": "Slot 'Slot_3' berhasil ditambahkan.",
  "slot": {
    "slot_id": "Slot_3",
    "polygon": [[34, 657], [158, 657], [238, 843], [79, 846]]
  }
}
```

Jika `slot_id` sudah ada, mengembalikan `409 Conflict`.

#### 4. Perbarui Satu Slot (`PUT /api/v1/slots/{slot_id}`)

```bash
curl -X PUT "http://127.0.0.1:8000/api/v1/slots/Slot_1" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "slot_id": "Slot_1",
    "polygon": [[1330, 330], [1450, 330], [1570, 480], [1430, 485]]
  }'
```

Response (`200 OK`):

```json
{
  "success": true,
  "message": "Slot 'Slot_1' berhasil diperbarui.",
  "slot": {
    "slot_id": "Slot_1",
    "polygon": [[1330, 330], [1450, 330], [1570, 480], [1430, 485]]
  }
}
```

#### 5. Hapus Satu Slot (`DELETE /api/v1/slots/{slot_id}`)

```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/slots/Slot_1" \
  -H "X-API-Key: $API_KEY"
```

Response (`200 OK`):

```json
{
  "success": true,
  "message": "Slot 'Slot_1' berhasil dihapus."
}
```

#### 6. Bulk Overwrite Seluruh Slot (`PUT /api/v1/slots`)

```bash
curl -X PUT "http://127.0.0.1:8000/api/v1/slots" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "slots": [
      {
        "slot_id": "Slot_1",
        "polygon": [[1336, 333], [1448, 332], [1568, 482], [1428, 486]]
      }
    ]
  }'
```

Response (`200 OK`):

```json
{
  "success": true,
  "message": "1 slot berhasil disimpan."
}
```

#### 7. Hapus Semua Slot (`DELETE /api/v1/slots`)

```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/slots" \
  -H "X-API-Key: $API_KEY"
```

Response (`200 OK`):

```json
{
  "success": true,
  "message": "Semua slot berhasil dihapus.",
  "slots": []
}
```


---

## Outbound Webhook

Webhook dikirim setiap kali status slot berubah (`EMPTY` ↔ `OCCUPIED`). Pengiriman dilakukan asynchronous melalui queue, sehingga tidak memperlambat response `/detect`.

Aktifkan dengan:

```env
WEBHOOK_URL=http://localhost:8080/webhook/parking
WEBHOOK_API_KEY=secret-key
WEBHOOK_TIMEOUT=5
WEBHOOK_RETRIES=3
```

Header request:

```http
Content-Type: application/json
X-API-Key: secret-key
```

Payload:

```json
{
  "timestamp": "2026-10-26T17:27:00Z",
  "event": "slot_state_changed",
  "summary": {
    "total_slots": 16,
    "occupied_slots": 4,
    "empty_slots": 12
  },
  "slots": [
    {
      "slot_id": "Slot_1",
      "is_occupied": true,
      "confidence": 0.92
    }
  ],
  "changed_slots": [
    {
      "slot_id": "Slot_1",
      "previous_is_occupied": false,
      "is_occupied": true,
      "confidence": 0.92
    }
  ]
}
```

- `slots` berisi snapshot lengkap semua slot.
- `changed_slots` hanya berisi slot yang berubah pada frame tersebut.
- Webhook hanya dikirim jika `changed_slots` tidak kosong.
- Jika `WEBHOOK_API_KEY` kosong, service memakai `API_KEY`.
- Retry memakai exponential backoff; kegagalan dicatat di log tanpa menggagalkan request API.

Contoh receiver sederhana (Go/Express/FastAPI) harus mengembalikan HTTP 2xx agar dianggap sukses.

---

## Panduan Integrasi Web UI & Backend

### 1. Pola Arsitektur Integrasi

Terdapat dua pola integrasi yang didukung:

#### Pola A: Web UI via Go Backend (Rekomendasi Produksi)

```text
Web UI (Browser)
  │  Polling status tiap 1-2 detik / WebSocket
  ▼
Go Backend
  │  POST /api/v1/detect        (stream frame CCTV + X-API-Key)
  │  GET  /api/v1/slots/status  (ambil snapshot status)
  │  GET  /api/v1/slots         (ambil polygon sekali saat inisialisasi)
  │  ◄── Webhook slot_state_changed (event-driven real-time push)
  ▼
FastAPI Vision Service
```

Keuntungan:

- `API_KEY` tersimpan aman di Go backend, tidak pernah terekspos ke browser client.
- Go backend dapat meng-cache response atau memancarkan perubahan slot ke Web UI secara instan via WebSocket / Server-Sent Events (SSE).

#### Pola B: Web UI Langsung ke Vision Service (Development / Dashboard Internal)

```text
Web UI (React / Vite / Vue @ localhost:5173 / localhost:3000)
  │  Fetch API langsung dengan header X-API-Key
  ▼
FastAPI Vision Service (dengan CORS diaktifkan: CORS_ORIGINS=*)
```

Keuntungan:

- Frontend developer dapat langsung melakukan *mocking*, membuat canvas editor slot, atau menampilkan live dashboard tanpa harus menjalankan backend Go terlebih dahulu.

---

### 2. Sistem Koordinat & Responsif Rendering

Koordinat vertex slot parkir (`polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`) disimpan dalam **ruang piksel native video/CCTV** (misalnya `1920x1080` atau `1280x720`).

Saat Web UI merender polygon di atas elemen `<video>`, `<canvas>`, atau `<svg>` yang ukurannya responsif (CSS fluid width):

#### Rumus Transformasi Skala

```javascript
const scaleX = containerWidth / originalVideoWidth;   // mis. 960 / 1920 = 0.5
const scaleY = containerHeight / originalVideoHeight; // mis. 540 / 1080 = 0.5

const screenX = point[0] * scaleX;
const screenY = point[1] * scaleY;
```

#### Contoh Rendering dengan HTML5 Canvas

```javascript
function drawSlot(ctx, slot, isOccupied, scaleX, scaleY) {
  const points = slot.polygon.map(([x, y]) => [x * scaleX, y * scaleY]);

  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.closePath();

  // Warna semantik: Merah jika terisi, Hijau jika kosong
  ctx.fillStyle = isOccupied ? "rgba(239, 68, 68, 0.4)" : "rgba(34, 197, 94, 0.35)";
  ctx.strokeStyle = isOccupied ? "#ef4444" : "#22c55e";
  ctx.lineWidth = 2;

  ctx.fill();
  ctx.stroke();

  // Label Slot
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText(slot.slot_id, points[0][0] + 4, points[0][1] + 16);
}
```

---

### 3. Semantik Status & Penanganan Kamera Offline (*Staleness*)

Endpoint `GET /api/v1/slots/status` mengembalikan field `last_inference` yang merekam waktu frame terakhir diproses:

| Kondisi | Indikator Visual | Deskripsi |
|---|---|---|
| `is_occupied: false` | Hijau (`#22c55e`) | Slot kosong dan siap digunakan |
| `is_occupied: true` | Merah (`#ef4444`) | Slot terisi kendaraan |
| `last_inference === null` | Kuning / Banner | Belum ada frame yang dikirim sejak server start |
| `now - last_inference > 10s` | Abu-abu / Peringatan | Feed kamera macet / terputus (*Stale Feed*) |

#### Logika Pengecekan Staleness di Frontend

```javascript
function checkFeedHealth(lastInferenceIso) {
  if (!lastInferenceIso) {
    return { status: "WAITING_FEED", message: "Menunggu aliran frame CCTV..." };
  }
  const ageMs = Date.now() - new Date(lastInferenceIso).getTime();
  if (ageMs > 10000) {
    return { status: "OFFLINE", message: `Kamera offline (${Math.round(ageMs / 1000)} detik lalu)` };
  }
  return { status: "LIVE", message: "Feed CCTV Aktif" };
}
```


---

### 4. Definisi Tipe TypeScript (Siap Pakai untuk Frontend)

```typescript
export interface SlotCoordinate {
  x: number;
  y: number;
}

export interface SlotConfig {
  slot_id: string;
  polygon: [number, number][]; // 4 titik [[x, y], ...]
}

export interface SlotsListResponse {
  success: boolean;
  total: number;
  slots: SlotConfig[];
}

export interface SlotMutationResponse {
  success: boolean;
  message: string;
  slot?: SlotConfig;
  slots?: SlotConfig[];
}

export interface SlotStatus {
  slot_id: string;
  is_occupied: boolean;
  confidence: number;
}

export interface Summary {
  total_slots: number;
  occupied_slots: number;
  empty_slots: number;
}

export interface StatusPayload {
  timestamp: string;
  summary: Summary;
  slots: SlotStatus[];
  last_inference: string | null;
}

export interface WebhookPayload extends StatusPayload {
  event: "slot_state_changed";
  changed_slots: {
    slot_id: string;
    previous_is_occupied: boolean;
    is_occupied: boolean;
    confidence: number;
  }[];
}
```

---

### 5. Contoh Polling Hook (React / TypeScript)

```typescript
import { useState, useEffect } from "react";

export function useParkingStatus(apiUrl: string, apiKey: string, intervalMs = 2000) {
  const [data, setData] = useState<StatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: number;
    let isMounted = true;

    async function fetchStatus() {
      try {
        const res = await fetch(`${apiUrl}/api/v1/slots/status`, {
          headers: apiKey ? { "X-API-Key": apiKey } : {},
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json: StatusPayload = await res.json();
        if (isMounted) {
          setData(json);
          setError(null);
        }
      } catch (err: any) {
        if (isMounted) setError(err.message);
      } finally {
        if (isMounted) timer = window.setTimeout(fetchStatus, intervalMs);
      }
    }

    fetchStatus();
    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, [apiUrl, apiKey, intervalMs]);

  return { data, error };
}
```


---

## Confidence dan Status

- `is_occupied` adalah status final setelah temporal smoothing.
- `confidence` adalah confidence YOLO maksimum dari kendaraan yang match dengan slot.
- Slot kosong memiliki `confidence: 0.0`.
- Jika slot ditahan `OCCUPIED` karena temporal buffer (frame saat ini tidak match), confidence terakhir yang valid dipertahankan.
- Status akan berubah setelah `SLOT_STATE_FRAMES` observasi konsisten (default 5 frame).

---

## Model Weights dan Optimasi VRAM

| Varian | Contoh File | VRAM (approx.) | Kecepatan | Akurasi | Rekomendasi |
|---|---|---|---|---|---|
| Nano | `yolo26n.pt`, `yolov8n.pt` | ~1-2 GB | Tercepat | Rendah | CPU / prototyping |
| Small | `yolo26s.pt` | ~2-3 GB | Sangat cepat | Sedang | GPU entry-level |
| Medium | `yolo26m.pt` | ~4-5 GB | Cepat | Tinggi | RTX 3050/3060 4GB |
| X-Large | `yolo26x.pt` | ~8+ GB | Lambat | Tertinggi | GPU desktop |

Tuning:

| Parameter | Efek |
|---|---|
| `YOLO_IMGSZ=1280` | Recall objek kecil/jauh naik; biaya inference ~2-3x |
| `OCCUPANCY_IOA_THRESHOLD` | Turunkan jika kendaraan sering tidak terdeteksi di slot |
| `SLOT_STATE_FRAMES` | Naikkan untuk anti-flicker lebih kuat, turunkan untuk respons lebih cepat |
| `CLASS_CONF` | Ambang per-kelas di `app/detector.py` |
| `DEVICE=cpu` | Paksa CPU untuk debugging |

---

## Testing

```bash
python -m compileall app
python -m pytest
```

Test mencakup:

- Kesetaraan point-in-polygon vectorized dengan implementasi per-polygon.
- Kesetaraan IoA vectorized dengan referensi `cv2.intersectConvexConvex`.
- Perilaku temporal smoothing (anti-flicker, konfirmasi N frame, confidence hold).
- Endpoint health/status/detect dan autentikasi API key.
- Pengiriman webhook, header API key, dan retry.

---

## Demo Video

```bash
python test_video.py
```

Opsi:

```bash
python test_video.py --source test_videos/video1.mp4
python test_video.py --source 0
```

---

## Perubahan Breaking

- `POST /api/v1/detect` sekarang mengembalikan schema di top-level, bukan `{ "success": true, "data": ... }`.
- Field `is_occupied_raw` tidak lagi ada di response publik.
- Konfigurasi contoh berpindah dari `.envexample` ke `.env.example`.
- Endpoint `/api/v1/*` membutuhkan `X-API-Key` jika `API_KEY` di-set.

---

## Lisensi

Proyek ini berlisensi **Apache License 2.0** — lihat [LICENSE](LICENSE).
