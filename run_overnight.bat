@echo off
REM Activate virtual environment once
call venv\Scripts\activate

REM Loop 7 times
for /L %%i in (1,1,7) do (
    echo Starting iteration %%i at %date% %time% >> run_log.txt

    cd Research
    python vit_imagenet1k_attacking_will_edit_cauchy.py ./data/ILSVRC2012_5K >> ..\run_log.txt 2>&1
    cd ..

    echo Finished iteration %%i at %date% %time% >> run_log.txt
)
