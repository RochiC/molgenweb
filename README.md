# Chem SMILES Generator

Una API REST construida con FastAPI para generar moléculas SMILES usando el modelo ChemGPT-4.7M de Hugging Face.

## 🚀 Instalación y Configuración

### 1. Crear y activar entorno virtual

**Windows (PowerShell):**
```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Iniciar el servidor

```powershell
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Parámetros del servidor:**
- `--reload`: Reinicia automáticamente cuando cambias el código
- `--host 0.0.0.0`: Permite conexiones desde cualquier IP
- `--port 8000`: Puerto donde se ejecuta la aplicación

## 📋 Uso de la API

### Endpoints disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Verificar estado de la aplicación |
| POST | `/generate` | Generar moléculas SMILES |

### Acceso a la documentación

Una vez iniciado el servidor, puedes acceder a:

- **Aplicación**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc

## 🧪 Ejemplos de uso

### 1. Verificar estado (Health Check)

```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "device": "cpu",
  "model": "ncfrey/ChemGPT-4.7M"
}
```

### 2. Generar SMILES

**Ejemplo básico:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "C"
  }'
```

**Ejemplo con parámetros personalizados:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "CCO",
    "max_length": 80,
    "top_k": 40,
    "top_p": 0.9,
    "temperature": 0.8
  }'
```

### Parámetros de generación

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `input_text` | string | - | Texto inicial para generar SMILES (requerido) |
| `max_length` | int | 60 | Longitud máxima de la secuencia generada |
| `top_k` | int | 50 | Número de tokens más probables a considerar |
| `top_p` | float | 0.95 | Probabilidad acumulativa para nucleus sampling |
| `temperature` | float | 1.0 | Controla la aleatoriedad (valores más bajos = más determinista) |

### Respuesta de ejemplo

```json
{
  "raw_tokens_string": "C[C@H](O)CO",
  "smiles_postprocesado": "C[C@H](O)CO"
}
```

## 🔧 Configuración

### Variables de entorno

Puedes configurar el modelo usando variables de entorno:

```bash
# Windows
set MODEL_NAME=ncfrey/ChemGPT-4.7M

# macOS/Linux
export MODEL_NAME=ncfrey/ChemGPT-4.7M
```

### Requisitos del sistema

- Python 3.8+
- 4GB+ RAM (recomendado para el modelo)
- GPU opcional (CUDA compatible para mejor rendimiento)

## 🛠️ Desarrollo

### Estructura del proyecto

```
├── main.py            # Aplicación principal FastAPI
├── requirements.txt   # Dependencias de Python
├── README.md          # Este archivo
├── render.yaml        # Configuración para deploy en Render
└── venv/              # Entorno virtual (después de crear)
```

### Comandos útiles

```bash
# Verificar dependencias instaladas
pip list

# Actualizar requirements.txt
pip freeze > requirements.txt

# Ejecutar en producción (sin --reload)
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Ejecutar en un puerto diferente
python -m uvicorn main:app --reload --port 8080
```

## 🐛 Solución de problemas

### Error: "Could not import module 'main'"
- Asegúrate de estar en el directorio correcto donde está `main.py`
- Verifica que el entorno virtual esté activado

### Error de memoria
- El modelo puede requerir bastante RAM. Cierra otras aplicaciones si es necesario
- Considera usar un modelo más pequeño si tienes limitaciones de hardware

### Puerto en uso
- Cambia el puerto: `--port 8001`
- O mata el proceso que usa el puerto 8000

## 📝 Licencia

Este proyecto está bajo la licencia MIT.