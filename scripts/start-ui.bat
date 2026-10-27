@echo off
rem pxconv local web UI launcher (double-click). Stdlib only, no Flask.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" goto :venv
start "pxconv-ui" /min python "ui\server.py" 8765
goto :eof
:venv
start "pxconv-ui" /min ".venv\Scripts\python.exe" "ui\server.py" 8765
goto :eof
