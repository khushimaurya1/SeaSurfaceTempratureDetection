# Sea Surface Temperature (SST) Detection — Web Frontend

This repository contains a Sea Surface Temperature (SST) detection demo with a web frontend.

What's included
- `seasurface.py` — Main SST detection logic (data generation, analysis, visualization).
- `app.py` — Flask frontend (full integration with `seasurface.py`).
- `app_simple.py` — Simplified Flask frontend that avoids heavy imports (recommended if you have import/runtime issues).
- `templates/index.html` — Web UI for interacting with the SST demo.
- `requirements.txt` — Python dependencies.

Quickstart (Windows)
1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

3. Run the simplified web server (recommended if you experienced heavy import delays):

```powershell
python app_simple.py
```

4. Or run the full frontend (if your environment has all packages and `seasurface.py` loads without long imports):

```powershell
python app.py
```

5. Open your browser and go to:

```
http://localhost:5000
```

Notes and troubleshooting
- If importing `seaborn` or `scipy` causes long delays or import-time errors on your machine, use `app_simple.py` which provides the same frontend but generates synthetic data without importing the full `seasurface.py` heavy modules.
- If `flask` script path warnings appear on Windows, ensure `C:\Users\<you>\AppData\Roaming\Python\Python<version>\Scripts` is on your PATH, or run via `python app.py`.
- To stop the server, press `Ctrl+C` in the terminal where it is running.

Next steps I can do for you
- Create a git branch and commit these changes.
- Add unit tests for core analysis functions in `seasurface.py`.
- Dockerize the app for local container testing.

If you'd like one of those next steps, tell me which and I will proceed.
