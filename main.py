import os
import re
import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Chem SMILES Generator", version="1.0.0")

# Configuración de CORS
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

# Add logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"Request: {request.method} {request.url}")
    print(f"Headers: {request.headers}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response

# ---------- Config ----------
MODEL_NAME = os.getenv("MODEL_NAME", "ncfrey/ChemGPT-4.7M")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"}

# Validación adicional
MIN_LENGTH = 10
MAX_LENGTH = 100

# ---------- Modelo global (cargado una vez) ----------
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

# ---------- Utilidades de decodificación/postpro ----------
def decodificar_tokens(tokens):
    mol = []
    for tok in tokens:
        if tok in SPECIAL_TOKENS:
            continue
        if tok.startswith("[") and tok.endswith("]"):
            contenido = tok[1:-1]
            if re.match(r'^[A-Za-z0-9@=#+\\/-]+$', contenido):
                mol.append(contenido)
            else:
                mol.append(tok)
        else:
            mol.append(tok)
    return "".join(mol)

def postprocesar_smiles(tokens_string: str) -> str:
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

            if tok.startswith("[Branch"):
                result.append("(")
                branch_stack.append(")")

            elif tok.startswith("Ring"):
                nums = re.findall(r'\d+', tok)
                if nums:
                    n = nums[0]
                    if n not in ring_open:
                        ring_open[n] = True
                    else:
                        del ring_open[n]
                    result.append(n)

            else:
                result.append(tok)

    while branch_stack:
        result.append(branch_stack.pop())

    return "".join(result)

# ---------- Tipos de request/response ----------
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

    def validate_params(self):
        if not self.input_text or len(self.input_text.strip()) == 0:
            raise HTTPException(
                status_code=422, 
                detail={
                    "error": "Validation Error",
                    "message": "input_text cannot be empty",
                    "field": "input_text"
                }
            )
        if self.max_length < MIN_LENGTH or self.max_length > MAX_LENGTH:
            raise HTTPException(
                status_code=422, 
                detail={
                    "error": "Validation Error",
                    "message": f"max_length must be between {MIN_LENGTH} and {MAX_LENGTH}",
                    "field": "max_length"
                }
            )
        if self.top_k < 1:
            raise HTTPException(
                status_code=422, 
                detail={
                    "error": "Validation Error",
                    "message": "top_k must be greater than 0",
                    "field": "top_k"
                }
            )
        if self.top_p <= 0 or self.top_p > 1:
            raise HTTPException(
                status_code=422, 
                detail={
                    "error": "Validation Error",
                    "message": "top_p must be between 0 and 1",
                    "field": "top_p"
                }
            )
        if self.temperature <= 0:
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

# ---------- Rutas ----------
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
            
        req.validate_params()
        
        if not model_loaded or tokenizer is None or model is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service Unavailable",
                    "message": "Modelo aún inicializando. Intente nuevamente en unos segundos."
                },
                headers=headers
            )

        inputs = tokenizer(req.input_text, return_tensors="pt").to(DEVICE)

        # Preferimos el token [EOS] si existe, si no el por defecto del tokenizer
        eos_id = tokenizer.convert_tokens_to_ids("[EOS]")
        if eos_id is None or eos_id == tokenizer.unk_token_id:
            eos_id = tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=req.max_length,
                do_sample=True,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
                eos_token_id=eos_id,
            )

        tokens = tokenizer.convert_ids_to_tokens(outputs[0])
        tokens_string = decodificar_tokens(tokens)
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

# Add OPTIONS endpoint
@app.options("/generate")
async def generate_options():
    return {"message": "OK"}
