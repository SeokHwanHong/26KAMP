@echo off
cd /d "%~dp0"
echo KAMP checks + CN7 RF setup (2026-10-05)
where python >/dev/null 2>nul
if errorlevel 1 goto USEPY
python "%~dp0run_20261005_checks_and_rf.py"
goto DONE
:USEPY
py "%~dp0run_20261005_checks_and_rf.py"
:DONE
echo.
echo DONE. Close this window and tell Claude.
pause
