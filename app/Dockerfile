FROM python:3.12-slim

WORKDIR /frontend

COPY app/requirements.txt /frontend/requirements.txt
RUN pip install --no-cache-dir -r /frontend/requirements.txt

COPY app/simple_direct_app.py /frontend/simple_direct_app.py

EXPOSE 8501

CMD ["streamlit", "run", "simple_direct_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
