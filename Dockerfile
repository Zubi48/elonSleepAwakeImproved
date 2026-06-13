FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY sleep_wake_conditional_probability_elon_v2.py .
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data
STOPSIGNAL SIGTERM
CMD ["python", "sleep_wake_conditional_probability_elon_v2.py"]
