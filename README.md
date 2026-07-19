<div align="center">

# BRUTUS

### An offline voice assistant that thinks on the edge. Built for Snapdragon X Elite.

You talk, it listens, reasons, and talks back. The speech recognition, the language model, and the voice all run on the Qualcomm NPU in the same laptop. No cloud round trip on the critical path. Turn the internet off and it still holds a conversation.

<br/>

[![Snapdragon X Elite](https://img.shields.io/badge/Snapdragon-X%20Elite-C41E3A?style=for-the-badge&logo=qualcomm&logoColor=white)](https://www.qualcomm.com/products/mobile/snapdragon/laptops-and-tablets/snapdragon-x-elite)
[![Hexagon NPU](https://img.shields.io/badge/Runs%20on-Hexagon%20NPU-6E4AFF?style=for-the-badge)](https://www.qualcomm.com/products/technology/processors/hexagon)
[![Qwen3 4B](https://img.shields.io/badge/LLM-Qwen3%204B%20(W4A16)-00A67E?style=for-the-badge)](https://huggingface.co/Qwen/Qwen3-4B)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12%20ARM64-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-EAB308?style=for-the-badge)](LICENSE)

</div>

---

## Team Brutus

| Name | Email |
|---|---|
| Aditya Pandey | aditya060806@gmail.com |
| Palak Rai | palakrai32323@gmail.com |
| Avik Srivastava | aviksrivastava786@gmail.com |

---

## Table of contents

* [Application description](#application-description)
* [Why this runs on the edge](#why-this-runs-on-the-edge)
* [Every model we use](#every-model-we-use)
* [Compression, or why 4 billion parameters fit](#compression-or-why-4-billion-parameters-fit)
* [Efficiency, and how it stays fast](#efficiency-and-how-it-stays-fast)
* [Architecture](#architecture)
* [The HTTP contract](#the-http-contract)
* [Setup from scratch](#setup-from-scratch)
* [Run and usage](#run-and-usage)
* [Tests and testing instructions](#tests-and-testing-instructions)
* [Platforms](#platforms)
* [Notes](#notes)
* [References](#references)
* [License](#license)

---

## Application description

Brutus is a private voice assistant that runs on a single Snapdragon X Elite laptop. Most assistants are a thin shell around a data center. You speak, your audio flies to a server you never see, a model you never chose answers, and a bill quietly grows. Brutus is the opposite. Speech becomes text, text becomes an answer, and the answer becomes speech, all on the Qualcomm silicon in the machine in front of you.

The product has two parts that talk over plain HTTP:

* **The Brain Node (this repository).** A headless inference appliance. It boots, loads three models once, holds them resident, and answers requests over the LAN forever. It runs no UI and no tools. It only does inference, and it does it on the Hexagon NPU. This is the piece that must run on the edge, and it does, end to end.
* **The Command PC cockpit.** A desktop app that captures your voice, draws a reactive face, runs a large tool layer, and routes every AI call to the Brain Node. It ships in a companion repository at [github.com/Aditya060806/Brutus](https://github.com/Aditya060806/Brutus). It performs no inference of its own. It is a window into a brain that lives on the network.

What you can actually do with it:

* Hold a spoken conversation that is transcribed, reasoned about, and answered fully on device.
* Type to it instead, through a text chat that routes to the same on device brain.
* Watch an on screen face react, because every reply begins with an emotion tag the UI reads to drive expression and lip movement.
* Use a wide tool layer from the cockpit: files and apps, OS automation, web search and research, email, presentations, a knowledge graph, and document question and answer.
* Flip a single switch between the on device voice engine and an optional cloud engine, for when no Brain Node is on the network.

---

## Why this runs on the edge

The hackathon asks that the majority of the application run locally on device. Here is exactly where every piece of compute happens.

| Component | Where it runs | On device |
|---|---|---|
| Speech to text (Whisper) | Snapdragon X Elite, ONNX Runtime | yes |
| Language model (Qwen3 4B) | Snapdragon X Elite, Hexagon NPU via GenieX | yes |
| Text to speech (Kokoro) | Snapdragon X Elite | yes |
| Cockpit app, tools, UI, face | Command PC, locally | yes |
| Vector memory, OCR | Command PC, locally | yes |
| Optional chat fallback and UI generation | Google Gemini (cloud) | no, optional only |

Every model in the voice loop runs on the edge device. The one path that can leave the machine is the optional cloud fallback, and it is off the critical path whenever a Brain Node is reachable. This is a hybrid design where the overwhelming majority runs locally, which is what edge first should mean in practice.

---

## Every model we use

Nothing hidden. This is the complete list, what each is, and where it executes.

### On device, in the voice loop (the Brain Node)

| Role | Model | Runtime | Device | Notes |
|---|---|---|---|---|
| LLM (default) | **Qwen3 4B**, quantized W4A16 | GenieX on QAIRT | Hexagon NPU | Ungated Qualcomm AI Hub bundle, about 3.0 GiB, roughly 15 tokens per second. The locked in default. |
| LLM (alt) | **Llama 3.2 3B Instruct SSD** | onnxruntime-genai, QNN EP | Hexagon NPU | Gated, needs a self export step. Selectable with `BRUTUS_LLM_BACKEND=genai`. |
| LLM (alt) | Any OpenAI compatible local server | AnythingLLM proxy | device | The fast path. Selectable with `BRUTUS_LLM_BACKEND=anythingllm`. |
| ASR | **Whisper base** | onnx-asr on ONNX Runtime | CPU execution provider | The ARM64 friendly path. Downloads from Hugging Face on first load. |
| TTS | **Kokoro v1.0** ONNX, voice `af_sarah` | kokoro-onnx | device | Fully offline, uses a bundled espeak-ng for phonemization. |

### In the cockpit (Command PC)

| Role | Model or engine | Where | Notes |
|---|---|---|---|
| Chat fallback | Google Gemini | cloud | Only used when no Brain Node is reachable. Optional. |
| UI generation | Google Gemini | cloud | Website and layout generation stays on the cloud model, since an edge 4B model is not the right tool for that. |
| Vector memory | LanceDB embeddings | device | Local document question and answer. |
| OCR | Tesseract | device | Reads text off the screen. |

### For development on any machine (no NPU)

| Role | Backend | Notes |
|---|---|---|
| LLM, ASR, TTS | `mock` | Deterministic fake backends, no heavy dependencies. Lets every team member build against the real HTTP contract on any laptop, and the whole test suite runs anywhere. |

The backend for each model type is chosen by a small factory (`app/backends/registry.py`) from config, so swapping any of them is a one line change.

---

## Compression, or why 4 billion parameters fit

A four billion parameter model at full precision is heavy. Quantization is what lets it sit on the NPU and still answer fast. Here is the weight footprint at three precisions.

```
Qwen3 4B weight footprint

  FP16    ████████████████████████████   ~8.0 GB   full precision baseline
  INT8    ██████████████                  ~4.0 GB   half of FP16
  W4A16   ██████████                      ~3.0 GB   what ships on the NPU
```

| Precision | Approx. weight size | vs FP16 | Where it runs |
|---|---|---|---|
| FP16 (baseline) | ~8.0 GB | 1.0x | GPU or big memory |
| INT8 | ~4.0 GB | 0.5x | reference point |
| **W4A16 (Brutus)** | **~3.0 GiB** | **~0.38x** | **Hexagon NPU** |

That is roughly **2.7x smaller** than the full precision baseline, a **62 percent** cut in footprint, without dropping to a tiny model that cannot hold a conversation. W4A16 keeps activations at sixteen bits, so quality stays high while the weights do the shrinking.

---

## Efficiency, and how it stays fast

The Brain Node is built to be boring in the best way. It starts once and then it is simply there.

| Property | How the node does it | Why it matters |
|---|---|---|
| Model loading | Loaded once at boot, held resident in `app/state.py` | No per request load cost, ever |
| Process model | Single worker | Models are never duplicated across processes |
| Cold start | A throwaway generation and TTS synth at boot | The first real request a judge triggers is already warm |
| NPU contention | GenieX isolated in its own process | Two QNN runtimes never fight over the NPU |
| Network | Served over the LAN, no internet | Works on a plane, in a lab, behind a firewall |
| Privacy | Nothing leaves the device | No audio or text is sent to a third party |
| LLM throughput | About 15 tokens per second on the Hexagon NPU | Comfortable for short spoken replies |

---

## Architecture

```
   You                Command PC cockpit                              Snapdragon X Elite  (Brain Node, this repo)
  ┌─────┐            ┌───────────────────────────────┐            ┌────────────────────────────────────┐
  │ 🎙️  │  ───────►  │  voice capture + avatar + UI  │  ───HTTP──►│  /asr    Whisper base   (CPU)        │
  │ 🗣️  │  ◄───────  │  tool layer + status pill     │  ◄───────  │  /v1/chat Qwen3 4B on the NPU        │
  └─────┘            └───────────────────────────────┘            │  /tts    Kokoro v1.0                 │
                        │  (optional cloud fallback)               │  /health status + heartbeat          │
                        ▼                                          └────────────────────────────────────┘
                     Google Gemini  (only when no Brain Node is reachable)
```

A full spoken turn on the on device engine:

```
  you speak
     │
     ▼  microphone capture and voice activity detection      (cockpit)
     ▼  POST /asr        Whisper base transcribes             (Brain Node, CPU)
     ▼  POST /v1/chat    Qwen3 4B answers on the NPU          (via GenieX)
     ▼  POST /tts        Kokoro v1.0 synthesizes the reply    (Brain Node)
     ▼  audio plays and the on screen face reacts             (cockpit)
```

The language model runs in a separate process called GenieX, Qualcomm's on device runtime on QAIRT, which exposes an OpenAI compatible server on port 18181. The Brain Node proxies to it. Keeping GenieX in its own process stops its QAIRT runtime from fighting this server's `onnxruntime-qnn` stack over the one NPU. The `start-brain-node.ps1` script starts both in the right order with one command.

The node is stateless. The client sends the full message array on every request and the node stores nothing between calls beyond an optional access log. That keeps the per device brain simple and fast, and leaves a clean seam for a shared memory layer later.

---

## The HTTP contract

The Brain Node exposes exactly four endpoints. That is the whole surface.

| Endpoint | Method | What it does |
|---|---|---|
| `/v1/chat` | POST | Chat from the LLM. Streams Server Sent Events, then a `brutus.metrics` event with time to first token and tokens per second. Shaped like the OpenAI API, so any OpenAI client works unchanged. |
| `/asr` | POST | Speech to text. Upload a WAV, get a transcript. |
| `/tts` | POST | Text to speech. Send text, get WAV audio back. |
| `/health` | GET | Liveness plus a capability report: execution providers, NPU status, warm up state, versions. Stays open even when a token is required, so dashboards can poll it. |

Every chat reply begins with an `[EMOTION:xxx]` tag, so the cockpit face and lip sync layer can drive expressions from it.

---

## Setup from scratch

There are two parts. You can run and demo the Brain Node completely on its own. The cockpit is the full voice experience on top.

### Prerequisites

* For the Brain Node: **Python 3.12**. On the Snapdragon device it must be the native **ARM64** build, or the NPU provider will not load.
* For real mode on device: a **Snapdragon X Elite** laptop and the **GenieX CLI**.
* For the cockpit: **Node.js 18 or newer** and **Windows 10 or 11**.

### Part A. The Brain Node (this repository)

Clone it and create a virtual environment.

```powershell
git clone https://github.com/Aditya060806/Ai-Qualcom-backend.git
cd Ai-Qualcom-backend

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` is small and pure Python: FastAPI, Uvicorn, Pydantic, python multipart, and httpx. Every wheel is prebuilt for Windows ARM64, so no compiler is needed. That alone is enough to run the node in mock mode on any machine.

For real inference on the Snapdragon device, add the on device dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-npu.txt
```

That pulls the confirmed working ARM64 wheels: `onnxruntime-qnn` (registers the NPU execution provider), `onnxruntime-genai`, `onnx-asr` for Whisper, and `kokoro-onnx` for the voice (which bundles espeak-ng, so nothing else to install).

Then bring up the language model on the NPU with the GenieX CLI:

```powershell
# install the GenieX CLI first: https://geniex.aihub.qualcomm.com/en/run/cli/install
$geniex = "C:\Users\<you>\AppData\Local\GenieX CLI\geniex.exe"
& $geniex pull ai-hub-models/Qwen3-4B --model-type llm    # about 3 GB, precompiled, no export
& $geniex serve                                           # OpenAI server on port 18181, on the NPU
```

### Part B. The Command PC cockpit (companion repository)

```bash
git clone https://github.com/Aditya060806/Brutus.git
cd Brutus
npm install
```

That is the whole install. Configuration is covered below.

---

## Run and usage

### Fastest path for a judge (any laptop, no NPU, about two minutes)

This proves the entire contract without special hardware.

```powershell
# in the Brain Node folder, with requirements.txt installed
.\scripts\run.ps1                     # serves 0.0.0.0:8080 in mock mode

# in a second terminal
.\.venv\Scripts\python.exe scripts\acceptance_test.py --base http://127.0.0.1:8080
```

You should see `/health` green, a WAV round trip through `/tts` then `/asr`, streamed chat tokens with a measured time to first token, and a final `== PASS ==`.

### Real, on the Snapdragon X Elite

1. Confirm native ARM64 Python: `python -c "import platform;print(platform.machine())"` prints `ARM64`.
2. Verify the NPU is visible: `.\.venv\Scripts\python.exe scripts\verify_npu.py` returns exit code 0 with `qnn_available: true`.
3. Start GenieX and the node together with one command:

```powershell
.\scripts\start-brain-node.ps1        # add -SkipServe if GenieX is already running
```

4. Re run `scripts\acceptance_test.py`. The transcript and speech are now real, and the chat reply streams from the NPU beginning with an `[EMOTION:...]` tag.

Check it from anywhere on the LAN:

```bash
curl http://<device ip>:8080/health
```

A healthy node reports `status: ok` with all three backends loaded.

### Talking to it through the cockpit

Run the cockpit and point it at the node.

```bash
npm run dev
```

Then in the app, open **Settings, then API Keys, then Brain Node** and set the URL, for example `http://<device ip>:8080`. Press **Save and Connect** and the status badge confirms the node is live. To choose where voice runs, open **Settings, then Voice Uplink**, and pick **Edge Server** for the on device loop or **Cloud** for the fallback. A small status pill shows the live state as it listens, thinks, and speaks.

Prefer environment variables for the cockpit? Set them in a `.env`:

```env
BRUTUS_BRAIN_URL="http://<device ip>:8080"   # where the Brain Node lives
BRUTUS_API_KEY=""                            # only if the node is locked with a token
GEMINI_API_KEY="your_key_here"               # optional cloud fallback and UI generation
```

### Direct API calls

```bash
# chat, streaming
curl -N http://<host>:8080/v1/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'

# text to speech
curl http://<host>:8080/tts -H "Content-Type: application/json" \
  -d '{"text":"hello"}' --output out.wav

# speech to text
curl http://<host>:8080/asr -F "file=@sample.wav"
```

---

## Tests and testing instructions

The node ships with a mock mode and a real acceptance gate, so you can verify the setup at two levels.

**Unit tests (mock mode, any machine):**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

These run against the deterministic mock backends, so they need no models and no NPU. They cover the HTTP contract for all four endpoints.

**End to end acceptance test:**

```powershell
.\.venv\Scripts\python.exe scripts\acceptance_test.py --base http://127.0.0.1:8080
```

This is the gate. It checks `/health`, does a real WAV round trip through `/tts` then `/asr`, streams a chat completion, and measures time to first token. A `== PASS ==` means the node is working. Run it in mock mode first, then again in real mode on the device to confirm the models.

**NPU check (device only):**

```powershell
.\.venv\Scripts\python.exe scripts\verify_npu.py
```

Exit code 0 with `qnn_available: true` confirms the Hexagon NPU execution provider is registered.

---

## Platforms

* **Brain Node:** Windows 11 on Snapdragon X Elite (ARM64) for real inference on the NPU. Runs in mock mode on any operating system with Python 3.12, for development and the plumbing tests.
* **Command PC cockpit:** Windows 10 and Windows 11. Packaged as a Windows installer with `npm run build:win`.

The application installs and runs from the instructions above, and behaves as described: a spoken turn is transcribed, answered, and spoken back, all on the edge device.

---

## Notes

* **Graceful degradation.** If a backend fails to load, the node logs it, marks it unavailable, and still boots serving the rest. If GenieX is not up, `/health` reports `degraded` and `/v1/chat` returns 503, while `/asr` and `/tts` keep working. The boot never crashes.
* **Two ways to bring up the LLM.** The recommended path is the ungated Qwen3 4B AI Hub bundle, which needs no export. The Llama 3.2 3B SSD path is available but is gated and requires a self export. During bring up you can also point the node at AnythingLLM or fall back to the mock LLM so ASR and TTS still demo.
* **GenieX gotchas we hit.** The model id in a chat request must omit the `:W4A16` precision suffix that the server advertises. The first request is a cold model load of about seventy seconds, then warm requests are seconds. Start GenieX before the node, since the node health checks it at boot.
* **Why onnx-asr and not faster-whisper.** faster-whisper depends on CTranslate2, which has no Windows ARM64 wheel. onnx-asr runs Whisper on ONNX Runtime and works on ARM64 today.
* **Quiet Qwen3.** Qwen3 emits verbose `<think>` blocks, so the node appends `/no_think` to the system prompt for clean, fast spoken replies. If you swap in a non Qwen model, clear `BRUTUS_LLM_SYSTEM_SUFFIX`.
* **Security.** By default the API is open on the LAN, which is fine on a closed network. Set `BRUTUS_API_KEY` to require a bearer token on every endpoint except `/health`. The cockpit stores keys locally through OS secure storage and sends no telemetry.
* **Stateless by design.** The node holds no session. Cross device memory, when it arrives, sits in front of it rather than inside it.
* **Request logging.** An optional transparent access log wrapper (`run-logged.ps1`) records one line per request plus a detailed JSON lines file, with binary bodies recorded by size only and sensitive headers redacted.

---

## References

* Qwen3 4B model card: <https://huggingface.co/Qwen/Qwen3-4B>
* Qualcomm AI Hub, browse compute models: <https://aihub.qualcomm.com/compute>
* GenieX CLI install and docs: <https://geniex.aihub.qualcomm.com/en/run/cli/install>
* ONNX Runtime QNN execution provider: <https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html>
* onnxruntime-genai: <https://github.com/microsoft/onnxruntime-genai>
* onnx-asr on PyPI: <https://pypi.org/project/onnx-asr/>
* OpenAI Whisper: <https://github.com/openai/whisper>
* Kokoro ONNX runtime: <https://github.com/thewh1teagle/kokoro-onnx>
* Kokoro 82M model: <https://huggingface.co/hexgrad/Kokoro-82M>
* espeak-ng: <https://github.com/espeak-ng/espeak-ng>
* FastAPI: <https://fastapi.tiangolo.com>
* Uvicorn: <https://www.uvicorn.org>
* Snapdragon X Elite: <https://www.qualcomm.com/products/mobile/snapdragon/laptops-and-tablets/snapdragon-x-elite>
* Qualcomm Hexagon NPU: <https://www.qualcomm.com/products/technology/processors/hexagon>
* Electron: <https://www.electronjs.org>
* React: <https://react.dev>
* Google Gemini API (optional cloud fallback): <https://ai.google.dev>

A note on the code itself: the source is documented where it matters. `config.py` explains every setting and the two runtime modes, `app/main.py` documents the boot lifespan step by step, and each backend in `app/backends/` states what it is and how it executes.

---

## License

MIT. See [LICENSE](LICENSE). Chosen with help from <https://choosealicense.com>, since it is permissive, simple, and lets anyone download, run, and build on Brutus.

<div align="center">
<br/>
<b>Brutus. Your voice, your machine, your model. Nothing leaves the room.</b>
</div>
