@echo off
cd /d "%~dp0"
echo KAMP check-only run (does not change runtime)
where python 1>NUL 2>NUL
if errorlevel 1 goto USEPY
python "%~dp0run_checks_only.py"
goto DONE
:USEPY
py "%~dp0run_checks_only.py"
:DONE
set RC=%ERRORLEVEL%
echo.
echo Exit code: %RC%  (0 = passed, 1 = checks failed, 2 = runner error)
echo DONE. Close this window and tell Claude.
pause
exit /b %RC%
