@echo off
echo Instalando dependencias...
pip install git+https://github.com/cvg/LightGlue.git
echo.
echo Iniciando servidor LightGlue...
python lightglue_server.py
pause