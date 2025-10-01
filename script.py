import os
import re
import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Chem SMILES Generator", version="1.0.0")

# ---------- Config ----------
MODEL_NAME = os.getenv("MODEL_NAME", "ncfrey/ChemGPT-4.7M")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"}

# ---------- Modelo global (cargado una vez) ----------
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
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

class GenerateResponse(BaseModel):
    raw_tokens_string: str
    smiles_postprocesado: str

# ---------- Rutas ----------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Modelo no inicializado")

    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando SMILES: {str(e)}")
