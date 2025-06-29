@echo on
REM Activate virtual environment once
call venv\Scripts\activate

REM Loop 7 times sequentially
for /L %%i in (1,1,10) do (
    echo Starting iteration %%i at %date% %time%

    cd Research
    python vit_imagenet1k_attacking_will_edit_cauchy.py ./data/ILSVRC2012_5K
    cd ..

    echo Finished iteration %%i at %date% %time%
)
