@echo off
set MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri file:./mlruns --port 5055 --workers 1