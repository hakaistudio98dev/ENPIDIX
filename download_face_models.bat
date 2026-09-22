@echo off
REM Unduh model wajah untuk ENPIDIX VMS (Windows).
REM Jalankan dari folder yang sama dengan server.py:  scripts\download_face_models.bat
setlocal enabledelayedexpansion
if "%FACE_MODEL_DIR%"=="" set FACE_MODEL_DIR=models
if not exist "%FACE_MODEL_DIR%" mkdir "%FACE_MODEL_DIR%"

set BASE=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models

call :get face_detection_yunet_2023mar.onnx  "%BASE%/face_detection_yunet/face_detection_yunet_2023mar.onnx"    232589
if errorlevel 1 goto :fail
call :get face_recognition_sface_2021dec.onnx "%BASE%/face_recognition_sface/face_recognition_sface_2021dec.onnx" 38696353
if errorlevel 1 goto :fail

echo.
echo Selesai. Restart server, buka /api/faces/engine/status untuk memastikan
echo "yunet": true dan "sface": true, lalu POST /api/faces/engine/retrain sekali.
exit /b 0

:get
set NAME=%~1
set URL=%~2
set SIZE=%~3
set OUT=%FACE_MODEL_DIR%\%NAME%
echo [unduh] %NAME% ...
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -Uri '%URL%' -OutFile '%OUT%' -UseBasicParsing } catch { Write-Host $_.Exception.Message; exit 1 }"
if errorlevel 1 exit /b 1
for %%A in ("%OUT%") do set ACTUAL=%%~zA
if not "!ACTUAL!"=="%SIZE%" (
  echo GAGAL: %NAME% berukuran !ACTUAL! byte, seharusnya %SIZE%.
  echo        Repo opencv_zoo memakai Git LFS - URL raw.githubusercontent hanya
  echo        mengembalikan pointer teks, bukan model. Kalau ini terus terjadi,
  echo        unduh manual lewat browser dan simpan ke %FACE_MODEL_DIR%\
  del "%OUT%" 2>nul
  exit /b 1
)
echo [ok]    %NAME%
exit /b 0

:fail
echo.
echo Pengunduhan gagal. Sistem TETAP JALAN tanpa model ini, tapi deteksi akan
echo memakai Haar cascade yang gagal pada wajah bermasker.
exit /b 1
