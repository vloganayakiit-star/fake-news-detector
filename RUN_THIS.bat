@echo off
echo ================================================
echo  FAKE NEWS DETECTOR - Maximum Accuracy Version
echo  By Gopika.R - Kongunadu Arts and Science
echo ================================================
echo.
echo [1] Installing libraries...
python -m pip install streamlit scikit-learn numpy pandas plotly --quiet
echo Done!
echo.
echo [2] Starting app...
echo.
echo  Browser opens at: http://localhost:8501
echo.
python -m streamlit run app.py
pause
