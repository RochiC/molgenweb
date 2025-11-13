import os
import re
import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Chem SMILES Generator", version="1.0.0")

# =========== CONFIGURACIÓN DE CORS ===========
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://mol-gen-ai.vercel.app",
    "https://molgenweb-ynuo.onrender.com",
    "https://www.mol-gen-ai.vercel.app",
    "*"  # Temporarily allow all origins in development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# =========== LOGGING MIDDLEWARE ===========
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"Request: {request.method} {request.url}")
    print(f"Headers: {request.headers}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response

# =========== CONFIG IA Y TOKENS ===========
MODEL_NAME = os.getenv("MODEL_NAME", "ncfrey/ChemGPT-4.7M")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"}

MIN_LENGTH = 10
MAX_LENGTH = 100

# =========== MODELO GLOBAL ==============
tokenizer = None
model = None
model_loaded = False

@app.on_event("startup")
def load_model():
    global tokenizer, model, model_loaded
    try:
        print(f"Cargando modelo {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        model_loaded = True
        print(f"Modelo cargado exitosamente en {DEVICE}")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        raise RuntimeError(f"No se pudo cargar el modelo '{MODEL_NAME}': {e}")

# ============= IA PROFESIONAL ==============
def decodificar_tokens(tokens):
    mol = []
    for tok in tokens:
        if tok in SPECIAL_TOKENS:
            continue
        # Chau corchetes solo de tokens especiales que no sean químicos tipo [C], [O], [N].
        if tok.startswith("[") and tok.endswith("]"):
            contenido = tok[1:-1]
            # Si el contenido es químico o token interno (Ring/Branch) lo mantiene o procesa
            if re.match(r'^[A-Za-z0-9@=#+\\/-]+$', contenido):
                mol.append(contenido)
            else:
                mol.append(tok)
        else:
            mol.append(tok)
    return "".join(mol)

def generar_smiles(input_text, max_length=60, top_k=50, top_p=0.95, temperature=1.0):
    global tokenizer, model
    if tokenizer is None or model is None:
        raise RuntimeError("Modelo no cargado")
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tokenizer.convert_tokens_to_ids("[EOS]") or tokenizer.eos_token_id,
    )
    tokens = tokenizer.convert_ids_to_tokens(outputs[0])
    tokens_string = decodificar_tokens(tokens)
    return tokens_string

def postprocesar_smiles(tokens_string):
    pattern = re.compile(r'\[.*?\]')
    tokens = pattern.split(tokens_string)
    matches = pattern.findall(tokens_string)

    result = []
    branch_stack = []
    ring_open = {}

    for i in range(len(tokens)):
        result.append(tokens[i])
        if i < len(matches):
            tok = matches[i]
            # Branch -> (
            if tok.startswith("[Branch"):
                result.append("(")
                branch_stack.append(")")
            # Ring -> manejar apertura/cierre
            elif tok.startswith("Ring"):
                num = re.findall(r'\d+', tok)
                if num:
                    n = num[0]
                    if n not in ring_open:
                        ring_open[n] = True
                    else:
                        del ring_open[n]
                    result.append(n)
            # Otros tokens (por seguridad)
            else:
                result.append(tok)
    while branch_stack:
        result.append(branch_stack.pop())
    return "".join(result)

# ============ REQUEST/RESPONSE TYPES ============
class GenerateRequest(BaseModel):
    input_text: str
    max_length: Optional[int] = 60
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    temperature: Optional[float] = 1.0

    class Config:
        schema_extra = {
            "example": {
                "input_text": "Generate a molecule similar to aspirin",
                "max_length": 60,
                "top_k": 50,
                "top_p": 0.95,
                "temperature": 1.0
            }
        }

# 🚩 Esta función va afuera de la clase, usa un parámetro "req" de tipo GenerateRequest
def validate_generate_request(req: GenerateRequest):
    if not req.input_text or len(req.input_text.strip()) == 0:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": "input_text cannot be empty",
                "field": "input_text"
            }
        )
    smiles_regex = re.compile(r'^[A-Za-z0-9@+\-=#%\/()\[\]\.\*]+$')
    if not smiles_regex.match(req.input_text):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": "Solo se permiten moléculas en formato SMILES (ej: CCO, C1CCCCC1, CC(=O)O, etc).",
                "field": "input_text"
            }
        )
    if req.max_length < MIN_LENGTH or req.max_length > MAX_LENGTH:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": f"max_length must be between {MIN_LENGTH} and {MAX_LENGTH}",
                "field": "max_length"
            }
        )
    if req.top_k < 1:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": "top_k must be greater than 0",
                "field": "top_k"
            }
        )
    if req.top_p <= 0 or req.top_p > 1:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": "top_p must be between 0 and 1",
                "field": "top_p"
            }
        )
    if req.temperature <= 0:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": "temperature must be greater than 0",
                "field": "temperature"
            }
        )

class GenerateResponse(BaseModel):
    raw_tokens_string: str
    smiles_postprocesado: str

# ============= ENDPOINTS =============
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_NAME,
        "model_loaded": model_loaded
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Content-Type": "application/json"
    }
    try:
        # Validate request body first
        if not isinstance(req, GenerateRequest):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid Request",
                    "message": "Invalid request format"
                },
                headers=headers
            )

        validate_generate_request(req)

        if not model_loaded or tokenizer is None or model is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service Unavailable",
                    "message": "Modelo aún inicializando. Intente nuevamente en unos segundos."
                },
                headers=headers
            )

        tokens_string = generar_smiles(
            req.input_text,
            req.max_length,
            req.top_k,
            req.top_p,
            req.temperature
        )
        smiles = postprocesar_smiles(tokens_string)
        return GenerateResponse(
            raw_tokens_string=tokens_string,
            smiles_postprocesado=smiles
        )

    except ValidationError as ve:
        print(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": str(ve)
            },
            headers=headers
        )
    except Exception as e:
        print(f"Error en generate: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": f"Error generando SMILES: {str(e)}"
            },
            headers=headers
        )

# OPTIONS endpoint
@app.options("/generate")
async def generate_options():
    return {"message": "OK"}
