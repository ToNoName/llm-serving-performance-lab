FROM python:3.11-slim

WORKDIR /app

COPY requirements/gateway.txt /tmp/gateway-requirements.txt
RUN pip install --no-cache-dir -r /tmp/gateway-requirements.txt

COPY src/gateway/ /app/gateway/

EXPOSE 8000

CMD ["uvicorn", "gateway.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
