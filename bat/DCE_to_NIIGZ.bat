@echo off
set script_dir=%~dp0
set python=%~dp0\python\python.exe
set input_path=%1
set datetime=%date:/=%%time::=%
set datetime=%datetime: =0%
set output_path=%script_dir%\%datetime%

if [%1]==[] goto usage

echo %input_path%
echo %output_path%

:loop
if "%~1" == "" goto end
"%python%" "%script_dir%\dcm2seq.py" %1 "%output_path%" --description description.csv --verbose 1 --prefix Dyn_ --offset 1 --logdir "%script_dir%\logs" --ext .nii.gz

shift
goto loop

goto :eof

:usage
@echo Usage: %0 ^<dicom_root_dir^>
exit /B 1

:end
pause