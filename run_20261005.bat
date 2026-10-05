@echo off
cd /d "%~dp0"
echo KAMP checks + CN7 RF setup (2026-10-05). WARNING: may register RF in the real runtime.
echo For checks only use run_checks_only.bat
where python 1>NUL 2>NUL
if errorlevel 1 goto USEPY
python "%~dp0run_20261005_checks_and_rf.py"
goto DONE
:USEPY
py "%~dp0run_20261005_checks_and_rf.py"
:DONE
set RC=%ERRORLEVEL%
echo.
echo Exit code: %RC%  (0 = ok, 1 = checks failed, 3 = RF/baseline problem, 2 = runner error)
echo DONE. Close this window and tell Claude.
pause
exit /b %RC%
