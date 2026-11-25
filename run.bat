@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Launching XAI MEDSCAN Streamlit Application...
"%PYTHON_EXE%" -m streamlit run app/streamlit_app.py
endlocal

