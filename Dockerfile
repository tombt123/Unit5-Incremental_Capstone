# Container for the churn prediction API.
#   docker build -t churn-api .
#   docker run -p 8000:8000 -e API_KEY=choose-a-secret churn-api
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY data/raw ./data/raw
COPY models ./models

# If no trained model was copied in, train one during the build so the image always works
# (and the scikit-learn version inside the image matches the one that trained the model).
RUN [ -f models/model.joblib ] || (python -m src.prepare_data && python -m src.train --trigger docker-build)

ENV API_KEY=change-me
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
