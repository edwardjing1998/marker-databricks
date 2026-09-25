FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api ./api
COPY shared ./shared
# OpenShift assigns an arbitrary nonroot UID. Code is read-only and /tmp is writable.
RUN chmod -R a+rX /app
USER 1001
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
