@echo off
echo Starting RUL Dashboard...
cd /d "%~dp0"

if not exist node_modules (
    echo Installing dashboard dependencies...
    npm install
)

:restart
node server.js
echo RUL Dashboard stopped. Restarting...
goto restart
