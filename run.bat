@echo off
REM Activate virtual environment
call venv\Scripts\activate

REM Run the Python script
cd Research
python vit_imagenet1k_attacking_will_edit_cauchy.py ./data/ILSVRC2012_5K

REM Optional: pause to see output
pause