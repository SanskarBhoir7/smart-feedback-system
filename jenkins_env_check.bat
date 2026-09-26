@echo off
REM ===========================================================================
REM  Jenkins runtime environment check
REM ===========================================================================
REM
REM  Add this as a Jenkins build step to confirm the Jenkins runtime can reach
REM  everything the pipeline needs, BEFORE debugging the pipeline itself:
REM
REM      New Item -> Freestyle project -> Build Steps
REM      -> Execute Windows batch command -> call jenkins_env_check.bat
REM
REM  It is deliberately written as a plain .bat so it behaves identically
REM  whether it is run by Jenkins, by cmd.exe, or by hand.
REM
REM  Which Windows account runs Jenkins matters: it decides which PATH,
REM  which Python and which Docker access the pipeline gets. On this machine
REM  the Jenkins service runs as LocalSystem and that WORKS - the
REM  ML-Deployment-Exp9 job ran `docker build` successfully as LocalSystem,
REM  and this project's Jenkins builds found python and git the same way.
REM  So do not assume a particular account is broken; run this script and
REM  let the checks below say what is actually missing.
REM ===========================================================================

setlocal
set FAILURES=0

echo ==========================================================
echo  JENKINS RUNTIME ENVIRONMENT CHECK
echo ==========================================================
echo   Workspace   : %CD%
echo   Running as  : %USERDOMAIN%\%USERNAME%
echo   Computer    : %COMPUTERNAME%
echo.

REM --- 1. python ------------------------------------------------------------
echo [1/7] python --version
python --version
if errorlevel 1 (
    echo       FAILED - python is not on the Jenkins PATH.
    echo       A Python installed "for this user only" is on that user's
    echo       PATH, not necessarily the service account's. Either install
    echo       Python for all users, or add its folder to the SYSTEM PATH
    echo       and restart the Jenkins service.
    set /a FAILURES+=1
) else (
    echo       OK
)
echo.

REM --- 2. git ---------------------------------------------------------------
echo [2/7] git --version
git --version
if errorlevel 1 (
    echo       FAILED - git is not on the Jenkins PATH.
    set /a FAILURES+=1
) else (
    echo       OK
)
echo.

REM --- 3. docker ------------------------------------------------------------
echo [3/7] docker --version
docker --version
if errorlevel 1 (
    echo       FAILED - docker CLI is not on the Jenkins PATH.
    set /a FAILURES+=1
) else (
    echo       OK
)
echo.

echo [3b ] docker ps  ^(daemon access - the usual failure point^)
docker ps
if errorlevel 1 (
    echo       FAILED - the CLI works but the daemon is unreachable.
    echo       Diagnose in this order:
    echo         a^) Is Docker Desktop actually running? Its engine can
    echo            stop on its own - `docker desktop start` restarts it.
    echo         b^) Can the account shown under "Running as" open the
    echo            Docker named pipe? Administrators and SYSTEM normally
    echo            can; any other account must be in 'docker-users'.
    echo         c^) Restart the Jenkins service after any group change -
    echo            group membership is only read at logon.
    set /a FAILURES+=1
) else (
    echo       OK
)
echo.

REM --- 4. persistent data directory ----------------------------------------
echo [4/7] access C:\mlops-data
if exist "C:\mlops-data" (
    dir /b "C:\mlops-data"
    echo       directory listing OK
) else (
    echo       FAILED - C:\mlops-data does not exist.
    set /a FAILURES+=1
)

echo       write test...
echo jenkins-write-test > "C:\mlops-data\.jenkins_write_test"
if errorlevel 1 (
    echo       FAILED - Jenkins cannot WRITE to C:\mlops-data.
    echo       The pipeline must write feedback.db and model_vN.pkl there.
    set /a FAILURES+=1
) else (
    del "C:\mlops-data\.jenkins_write_test" >nul 2>&1
    echo       OK - read and write
)
echo.

REM --- 5. the sentence encoder ----------------------------------------------
echo [5/7] python prepare_encoder.py --verify
python prepare_encoder.py --verify
if errorlevel 1 (
    echo       FAILED - the sentence encoder is missing or damaged in
    echo       C:\mlops-data\encoders. Run `python prepare_encoder.py` once
    echo       with internet access; after that no step needs the network.
    set /a FAILURES+=1
) else (
    echo       OK
)
echo.

REM --- 6. the retraining check (exit-code contract) -------------------------
echo [6/7] python retrain.py --check
python retrain.py --check
set RC=%ERRORLEVEL%
echo       exit code = %RC%
if "%RC%"=="0"  echo       OK - retraining NOT required
if "%RC%"=="10" echo       OK - retraining IS required
if not "%RC%"=="0" if not "%RC%"=="10" (
    echo       FAILED - expected exit code 0 or 10, got %RC%
    set /a FAILURES+=1
)
echo.

REM --- 7. tests and model staging ------------------------------------------
echo [7/7] pytest -q
python -m pytest -q
if errorlevel 1 (
    echo       FAILED - the test suite did not pass.
    set /a FAILURES+=1
) else (
    echo       OK
)
echo.

echo [7b ] python stage_model.py --clean
python stage_model.py --clean
if errorlevel 1 (
    echo       FAILED - could not stage the production model and encoder for Docker.
    set /a FAILURES+=1
) else (
    echo       OK
)
echo.

REM --- verdict --------------------------------------------------------------
echo ==========================================================
if "%FAILURES%"=="0" (
    echo  ALL CHECKS PASSED - the Jenkins runtime is ready.
    echo ==========================================================
    endlocal
    exit /b 0
) else (
    echo  %FAILURES% CHECK^(S^) FAILED - fix these before running the pipeline.
    echo ==========================================================
    endlocal
    exit /b 1
)
