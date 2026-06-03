@echo off
REM run_pipeline.bat — launcher for the PIV post-processing pipeline.
REM Drop this anywhere; double-click won't work but it can be called from
REM any folder via its full path, e.g.:
REM   "C:\...\focused-kilby-9a7427\run_pipeline.bat" --input-dir "..." --skip-masking
REM
REM Handles the cd-into-worktree dance and the python -m incantation for you.
python "%~dp0piv_pipeline\piv_pipeline_master.py" %*
