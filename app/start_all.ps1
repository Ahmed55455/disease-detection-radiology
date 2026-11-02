# Starts both servers for the demo (run from the repository root):
#   powershell -ExecutionPolicy Bypass -File app\start_all.ps1
# 1) FastAPI on port 8000  -> phone page:  http://<laptop-ip>:8000/mobile
# 2) Streamlit on port 8501 -> laptop web: http://localhost:8501
# Close the two new windows (or press Ctrl+C in them) to stop.
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$py' -m uvicorn app.api:app --host 0.0.0.0 --port 8000"
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$py' -m streamlit run app/app.py"
Write-Host "Started. Phone page: open the 'Phone capture' box in the app sidebar and scan the QR code."
