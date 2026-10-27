"""
SINGLE-FILE BUILD of the Urdu AI Video Maker backend for one-click hosting
(Hugging Face Spaces / any Docker host). Generated from backend/app/* —
logic identical; only the module packaging is flattened.
"""
import os
from dataclasses import dataclass
import json
import re
import httpx
import asyncio
import base64
import time
import uuid
import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ================= config =================
"""Environment-based configuration. API keys live ONLY here (from .env / env vars),
never in source code and never in the Android APK."""
import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    # Gemini (text) — free tier key from Google AI Studio
    gemini_api_key: str = _env("GEMINI_API_KEY")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")

    # Video provider: none | huggingface | veo | selfhost
    video_provider: str = _env("VIDEO_PROVIDER", "none").lower()

    # Hugging Face (free tier, limited) — owner's own free token
    hf_token: str = _env("HF_TOKEN")
    hf_video_model: str = _env("HF_VIDEO_MODEL", "Lightricks/LTX-Video")

    # Veo (PAID per second — only the backend owner may enable this with their
    # own billed key; the app itself never charges end users anything)
    veo_model: str = _env("VEO_MODEL", "veo-3.1-lite-generate-preview")

    # Self-hosted open-model server (e.g. Xinference on your own GPU)
    selfhost_url: str = _env("SELFHOST_URL", "http://127.0.0.1:9997")
    selfhost_model: str = _env("SELFHOST_MODEL", "Wan2.1-1.3B")

    # Storage for generated files
    data_dir: str = _env("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def video_configured(self) -> bool:
        if self.video_provider == "huggingface":
            return bool(self.hf_token)
        if self.video_provider == "veo":
            return bool(self.gemini_api_key)
        if self.video_provider == "selfhost":
            return bool(self.selfhost_url)
        return False


settings = Settings()
os.makedirs(settings.data_dir, exist_ok=True)


# ================= gemini =================
"""Google Gemini text client (free tier). Used ONLY for text work:
scripts, prompt enhancement, scene plans, titles, subtitle splitting.
It is never presented as a video generator."""
import json
import re

import httpx


BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiNotConfigured(Exception):
    pass


class GeminiError(Exception):
    pass


async def _generate(prompt: str, *, json_mode: bool = False, timeout: float = 120.0) -> str:
    if not settings.gemini_configured:
        raise GeminiNotConfigured(
            "Gemini API key is not set on the backend. Add GEMINI_API_KEY to backend/.env "
            "(free key: Google AI Studio) and restart the backend."
        )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7},
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    url = f"{BASE}/{settings.gemini_model}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=body, headers={"x-goog-api-key": settings.gemini_api_key})
    except httpx.TimeoutException:
        raise GeminiError("Gemini request timed out. Please try again.")
    except httpx.HTTPError as e:
        raise GeminiError(f"Network error talking to Gemini: {e}")
    if r.status_code == 400:
        raise GeminiError("Gemini rejected the request (400). Check the prompt and try again.")
    if r.status_code in (401, 403):
        raise GeminiError("Gemini API key was rejected (401/403). Check GEMINI_API_KEY on the backend.")
    if r.status_code == 429:
        raise GeminiError("Gemini free-tier rate limit reached (429). Wait a minute and retry.")
    if r.status_code >= 400:
        raise GeminiError(f"Gemini error {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        raise GeminiError("Gemini returned an empty response (possibly a safety filter). Rephrase and retry.")


def _parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if not m:
            raise GeminiError("Could not parse Gemini JSON output.")
        return json.loads(m.group(1))


LANG_NAME = {"ur": "Urdu (اردو)", "en": "English"}


async def make_script(topic: str, language: str, duration_sec: int, tone: str) -> str:
    words = max(20, int(duration_sec * 2.2))  # ~2.2 words/sec narration pace
    prompt = (
        f"Write a voice-over narration script in {LANG_NAME.get(language, 'Urdu')} for a {duration_sec}-second "
        f"short video about: {topic}. Tone: {tone}. About {words} words. "
        "Return ONLY the script text, no headings, no scene notes, no markdown."
    )
    return await _generate(prompt)


async def enhance_prompt(user_prompt: str, style: str) -> str:
    prompt = (
        "Improve this video-generation prompt for a text-to-video AI model. Keep the same subject and meaning, "
        f"add cinematic visual detail, lighting, camera movement. Style: {style}. "
        "Reply with ONLY the improved prompt in English (video models work best with English prompts).\n\n"
        f"Prompt: {user_prompt}"
    )
    return await _generate(prompt)


async def plan_scenes(script: str, language: str) -> list:
    prompt = (
        "Split this narration script into scenes for a short video. Return JSON: a list of objects with "
        '"title" (short, in the script language), "visual_prompt" (detailed English prompt for a video model), '
        '"duration_sec" (number, 4-8), "narration" (the exact part of the script for this scene). '
        f"The narrations together must cover the whole script.\n\nScript:\n{script}"
    )
    data = _parse_json(await _generate(prompt, json_mode=True))
    if isinstance(data, dict):
        data = data.get("scenes", [])
    return data


async def make_titles(topic: str, language: str) -> list:
    prompt = (
        f"Give 6 catchy short video titles in {LANG_NAME.get(language, 'Urdu')} for a video about: {topic}. "
        'Return JSON: a list of strings only.'
    )
    data = _parse_json(await _generate(prompt, json_mode=True))
    return [str(t) for t in data][:6]


async def split_subtitle_lines(script: str) -> list:
    """Ask Gemini to split the script into short subtitle lines (timing is computed in code)."""
    prompt = (
        "Split this narration into short subtitle lines (max ~42 characters each, break at natural pauses). "
        'Return JSON: a list of strings only.\n\n' + script
    )
    data = _parse_json(await _generate(prompt, json_mode=True))
    return [str(t).strip() for t in data if str(t).strip()]


# ================= providers =================
"""Pluggable video-generation providers + in-memory job store.

Honesty rules baked in:
- provider "none" (default) -> jobs fail immediately with a clear NOT_CONFIGURED
  message. No placeholder/fake video is ever produced.
- "veo" is PAID per second and only runs if the backend OWNER configured it with
  their own billed Gemini key. The Android app never charges end users.
- "huggingface" uses the owner's free HF token (free tier = small monthly
  credits + rate limits; good for testing, not unlimited).
- "selfhost" calls an open-model server (e.g. Xinference) on the owner's GPU.
"""
import asyncio
import base64
import os
import time
import uuid

import httpx


JOBS: dict = {}


def new_job(params: dict) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "status": "queued",  # queued | running | done | error
        "progress": "Queued",
        "params": params,
        "file": None,
        "error": None,
        "error_code": None,
        "created_at": time.time(),
    }
    JOBS[job["id"]] = job
    return job


class ProviderError(Exception):
    def __init__(self, message: str, code: str = "PROVIDER_ERROR"):
        super().__init__(message)
        self.code = code


def _out_path(job_id: str) -> str:
    return os.path.join(settings.data_dir, f"video_{job_id}.mp4")


async def _save_bytes(job: dict, content: bytes):
    if len(content) < 20_000:  # too small to be a real video -> likely an error body
        raise ProviderError(f"Provider returned {len(content)} bytes, not a video. Body: {content[:200]!r}")
    path = _out_path(job["id"])
    with open(path, "wb") as f:
        f.write(content)
    job["file"] = path


# ---------------------------------------------------------------- Hugging Face
async def _hf_generate(job: dict, params: dict):
    """HF Inference API for text-to-video models (e.g. Lightricks/LTX-Video).
    Sends the prompt; many T2V endpoints answer with raw MP4 bytes.
    Image-to-video payload shape varies by model — errors are surfaced as-is."""
    if not settings.hf_token:
        raise ProviderError("HF_TOKEN is not set on the backend.", "NOT_CONFIGURED")
    url = f"https://api-inference.huggingface.co/models/{settings.hf_video_model}"
    headers = {"Authorization": f"Bearer {settings.hf_token}"}
    payload = {"inputs": params["prompt"]}
    if params.get("mode") == "i2v" and params.get("image_base64"):
        payload = {"inputs": {"prompt": params["prompt"], "image": params["image_base64"]},
                   "parameters": {}}
    deadline = time.time() + 900  # free shared GPUs can be slow / cold
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        while True:
            job["progress"] = "Generating on Hugging Face (free tier — may wait for a GPU)…"
            r = await client.post(url, json=payload, headers=headers)
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and ("video" in ctype or "octet" in ctype):
                await _save_bytes(job, r.content)
                return
            if r.status_code == 503:  # model loading
                est = 20
                try:
                    est = int(r.json().get("estimated_time", 20))
                except Exception:
                    pass
                if time.time() > deadline:
                    raise ProviderError("Hugging Face model stayed busy/loading for 15 minutes. Try again later.", "TIMEOUT")
                job["progress"] = f"Model is loading on HF servers (~{est}s)…"
                await asyncio.sleep(min(est, 30))
                continue
            if r.status_code == 429:
                raise ProviderError("Hugging Face free-tier rate limit / monthly credits reached (429). "
                                    "Wait, or switch VIDEO_PROVIDER in backend/.env.", "RATE_LIMIT")
            if r.status_code in (401, 403):
                raise ProviderError("HF_TOKEN rejected (401/403). Check the token on the backend.", "AUTH")
            raise ProviderError(f"Hugging Face error {r.status_code}: {r.text[:300]}")


# ----------------------------------------------------------------------- Veo
async def _veo_generate(job: dict, params: dict):
    """Google Veo via Gemini API long-running operations. PAID per second —
    billed to the backend owner's Google account. Max 8 s per generation."""
    if not settings.gemini_api_key:
        raise ProviderError("GEMINI_API_KEY is not set on the backend (Veo needs it).", "NOT_CONFIGURED")
    key = settings.gemini_api_key
    model = settings.veo_model
    instance = {"prompt": params["prompt"]}
    if params.get("mode") == "i2v" and params.get("image_base64"):
        instance["image"] = {"bytesBase64Encoded": params["image_base64"], "mimeType": "image/jpeg"}
    body = {
        "instances": [instance],
        "parameters": {
            "aspectRatio": params.get("aspect", "9:16"),
            "durationSeconds": min(int(params.get("duration_sec", 8)), 8),
        },
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        job["progress"] = "Submitting to Veo…"
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predictLongRunning",
            json=body, headers={"x-goog-api-key": key})
        if r.status_code >= 400:
            raise ProviderError(f"Veo submit error {r.status_code}: {r.text[:300]}")
        op_name = r.json()["name"]
        # poll the operation
        for _ in range(120):  # up to ~10 minutes
            await asyncio.sleep(5)
            job["progress"] = "Veo is generating…"
            ro = await client.get(f"https://generativelanguage.googleapis.com/v1beta/{op_name}",
                                  headers={"x-goog-api-key": key})
            data = ro.json()
            if data.get("done"):
                if "error" in data:
                    raise ProviderError(f"Veo generation failed: {str(data['error'])[:300]}")
                try:
                    sample = data["response"]["generateVideoResponse"]["generatedSamples"][0]
                    uri = sample["video"]["uri"]
                except (KeyError, IndexError):
                    raise ProviderError("Veo finished but returned no video sample (possibly a safety filter).")
                rv = await client.get(uri, headers={"x-goog-api-key": key}, follow_redirects=True)
                await _save_bytes(job, rv.content)
                return
        raise ProviderError("Veo generation timed out after 10 minutes.", "TIMEOUT")


# ------------------------------------------------------------------ Self-host
async def _selfhost_generate(job: dict, params: dict):
    """Open-model server on the owner's own GPU (Xinference-compatible)."""
    url = settings.selfhost_url.rstrip("/") + "/v1/video/generations"
    body = {"model": settings.selfhost_model, "prompt": params["prompt"]}
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        job["progress"] = "Generating on your self-hosted GPU server…"
        r = await client.post(url, json=body)
        if r.status_code >= 400:
            raise ProviderError(f"Self-host server error {r.status_code}: {r.text[:300]}")
        ctype = r.headers.get("content-type", "")
        if "video" in ctype or "octet" in ctype:
            await _save_bytes(job, r.content)
            return
        data = r.json()
        video_url = None
        try:
            video_url = data["data"][0].get("url")
        except Exception:
            pass
        if not video_url:
            raise ProviderError(f"Self-host server returned an unexpected response: {str(data)[:300]}")
        rv = await client.get(video_url)
        await _save_bytes(job, rv.content)


PROVIDERS = {
    "huggingface": _hf_generate,
    "veo": _veo_generate,
    "selfhost": _selfhost_generate,
}


async def run_job(job: dict):
    params = job["params"]
    provider = settings.video_provider
    if provider == "none" or provider not in PROVIDERS:
        job["status"] = "error"
        job["error_code"] = "NOT_CONFIGURED"
        job["error"] = (
            "No video-generation provider is configured on the backend. "
            "Set VIDEO_PROVIDER in backend/.env to 'huggingface' (free, limited, needs HF_TOKEN), "
            "'selfhost' (your own GPU), or 'veo' (paid, owner's key). "
            "This app never creates fake videos."
        )
        return
    if not settings.video_configured:
        job["status"] = "error"
        job["error_code"] = "NOT_CONFIGURED"
        job["error"] = f"Provider '{provider}' is selected but its key/URL is missing in backend/.env."
        return
    job["status"] = "running"
    job["progress"] = "Starting…"
    try:
        await PROVIDERS[provider](job, params)
        job["status"] = "done"
        job["progress"] = "Done"
    except ProviderError as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["error_code"] = e.code
    except Exception as e:  # never crash the job silently
        job["status"] = "error"
        job["error"] = f"Unexpected error: {e}"
        job["error_code"] = "INTERNAL"


def validate_image_base64(b64: str) -> bytes:
    """Validate an uploaded image: real JPEG/PNG/WebP magic bytes, <= 10 MB."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        raise ProviderError("Image data is not valid base64.", "BAD_IMAGE")
    if len(raw) > 10 * 1024 * 1024:
        raise ProviderError("Image is larger than 10 MB.", "BAD_IMAGE")
    if raw[:3] == b"\xff\xd8\xff" or raw[:8] == b"\x89PNG\r\n\x1a\n" or raw[:4] == b"RIFF":
        return raw
    raise ProviderError("Only JPEG, PNG or WebP images are accepted.", "BAD_IMAGE")


# ================= module shims =================
import types as _types
gemini = _types.SimpleNamespace(GeminiError=GeminiError, GeminiNotConfigured=GeminiNotConfigured, enhance_prompt=enhance_prompt, make_script=make_script, make_titles=make_titles, plan_scenes=plan_scenes, split_subtitle_lines=split_subtitle_lines)
providers = _types.SimpleNamespace(JOBS=JOBS, new_job=new_job, run_job=run_job, validate_image_base64=validate_image_base64)


# ================= api =================
"""Free Urdu AI Video Maker — FastAPI backend.
Keys come from .env (see config.py). No payments, no user accounts, no tracking."""
import asyncio
import os

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


app = FastAPI(title="Free Urdu AI Video Maker Backend", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VOICES = {"male": "ur-PK-AsadNeural", "female": "ur-PK-UzmaNeural"}


# --------------------------------------------------------------------- models
class ScriptIn(BaseModel):
    topic: str = Field(min_length=2, max_length=2000)
    language: str = "ur"
    duration_sec: int = Field(default=30, ge=5, le=600)
    tone: str = "engaging"


class PromptIn(BaseModel):
    prompt: str = Field(min_length=2, max_length=2000)
    style: str = "Cinematic"


class ScenesIn(BaseModel):
    script: str = Field(min_length=2, max_length=8000)
    language: str = "ur"


class TitlesIn(BaseModel):
    topic: str = Field(min_length=2, max_length=500)
    language: str = "ur"


class SubtitlesIn(BaseModel):
    script: str = Field(min_length=2, max_length=8000)


class TtsIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str = "male"


class VideoJobIn(BaseModel):
    prompt: str = Field(min_length=2, max_length=2000)
    mode: str = "t2v"  # t2v | i2v
    aspect: str = "9:16"
    duration_sec: int = Field(default=8, ge=2, le=60)
    style: str = "Cinematic"
    image_base64: str | None = None


def _gemini_http(e: Exception) -> HTTPException:
    if isinstance(e, gemini.GeminiNotConfigured):
        return HTTPException(503, str(e))
    return HTTPException(502, str(e))


# --------------------------------------------------------------------- routes
@app.get("/health")
async def health():
    return {
        "ok": True,
        "gemini_configured": settings.gemini_configured,
        "gemini_model": settings.gemini_model,
        "video_provider": settings.video_provider,
        "video_configured": settings.video_configured,
        "tts_voices": VOICES,
    }


@app.post("/api/script")
async def script(body: ScriptIn):
    try:
        text = await gemini.make_script(body.topic, body.language, body.duration_sec, body.tone)
        return {"script": text}
    except (gemini.GeminiNotConfigured, gemini.GeminiError) as e:
        raise _gemini_http(e)


@app.post("/api/enhance-prompt")
async def enhance(body: PromptIn):
    try:
        return {"prompt": await gemini.enhance_prompt(body.prompt, body.style)}
    except (gemini.GeminiNotConfigured, gemini.GeminiError) as e:
        raise _gemini_http(e)


@app.post("/api/scenes")
async def scenes(body: ScenesIn):
    try:
        return {"scenes": await gemini.plan_scenes(body.script, body.language)}
    except (gemini.GeminiNotConfigured, gemini.GeminiError) as e:
        raise _gemini_http(e)


@app.post("/api/titles")
async def titles(body: TitlesIn):
    try:
        return {"titles": await gemini.make_titles(body.topic, body.language)}
    except (gemini.GeminiNotConfigured, gemini.GeminiError) as e:
        raise _gemini_http(e)


@app.post("/api/subtitles")
async def subtitles(body: SubtitlesIn):
    """Split the script into lines (Gemini), then time them deterministically:
    Urdu narration pace ~ 800 chars/min ≈ 13.3 chars/sec."""
    try:
        lines = await gemini.split_subtitle_lines(body.script)
    except (gemini.GeminiNotConfigured, gemini.GeminiError) as e:
        raise _gemini_http(e)
    cues, t = [], 0.0
    for line in lines:
        dur = max(1.2, len(line) / 13.3)
        cues.append({"start_ms": int(t * 1000), "end_ms": int((t + dur) * 1000), "text": line})
        t += dur + 0.15
    return {"cues": cues}


@app.post("/api/tts")
async def tts(body: TtsIn):
    """Urdu voice-over via Microsoft Edge TTS (free). Returns an MP3 file."""
    voice = VOICES.get(body.voice, VOICES["male"])
    out = os.path.join(settings.data_dir, f"tts_{abs(hash(body.text + voice))}.mp3")
    try:
        await edge_tts.Communicate(body.text, voice).save(out)
    except Exception as e:
        raise HTTPException(502, f"TTS failed: {e}. Check the backend's internet connection.")
    return FileResponse(out, media_type="audio/mpeg", filename="voiceover.mp3")


@app.post("/api/video/jobs")
async def create_video_job(body: VideoJobIn):
    params = body.model_dump()
    if body.mode == "i2v":
        if not body.image_base64:
            raise HTTPException(400, "Image-to-video needs an image.")
        providers.validate_image_base64(body.image_base64)
    job = providers.new_job(params)
    asyncio.create_task(providers.run_job(job))
    return {"job_id": job["id"], "status": job["status"]}


@app.get("/api/video/jobs/{job_id}")
async def job_status(job_id: str):
    job = providers.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (backend restarted?).")
    return {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
        "error_code": job["error_code"],
    }


@app.get("/api/video/jobs/{job_id}/file")
async def job_file(job_id: str):
    job = providers.JOBS.get(job_id)
    if not job or job["status"] != "done" or not job["file"]:
        raise HTTPException(404, "Video is not ready.")
    return FileResponse(job["file"], media_type="video/mp4", filename=f"urdu_ai_video_{job_id}.mp4")
