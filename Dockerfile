FROM python:3.10-slim

WORKDIR /app

# Instalar herramientas del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del agente y el documento PDF
COPY agent.py .
COPY futureapplication.pdf .

# Exponer el puerto predeterminado de FastAPI / LangGraph
EXPOSE 8000

# Comando para ejecutar la API del agente en producción
CMD ["langgraph", "api", "--host", "0.0.0.0", "--port", "8000"]