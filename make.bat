@echo off
setlocal enabledelayedexpansion

:: Verify if a command has been passed
if "%1"=="" (
    echo command: %1
    goto usage
)

:: Verify if the file has been passed
if "%2"=="" (
    echo file: %2
    echo Error: filename is obrigatory
)

set CMD=%1
set FILE=%2

:: Remove ext py
if /I "!FILE:~-3!" == ".py" (
    set FILE=!FILE:~0,-3!
)

if "!CMD!"=="run" (
    py -m models.!FILE! run
    exit /b 0
)

if "!CMD!"=="train" (
    py -m models.!FILE! train
    exit /b 0
)

echo Unknow command: !CMD!
goto usage



:: Use mode
:usage
echo Use: ./make (run, train) [file_name]
exit /b 1