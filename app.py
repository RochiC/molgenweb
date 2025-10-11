import gradio as gr
import torch
import main as core  # importa tu main.py

# ---------- Forzar carga del modelo al iniciar ----------
if core.tokenizer is None or core.model is None:
    print("🧠 Cargando modelo en Gradio (no vía FastAPI)...")
    core.load_model()


# ---------- Función de inferencia ----------
def run_inference(input_text: str):
    if not input_text.strip():
        return "Ingresá una configuración para generar el resultado."

    try:
        # Tokenización e inferencia
        inputs = core.tokenizer(input_text, return_tensors="pt").to(core.DEVICE)
        with torch.no_grad():
            outputs = core.model.generate(
        inputs["input_ids"],
        max_length=60,
        do_sample=True,          # activa muestreo aleatorio
        top_k=40,                # elige entre los 40 tokens más probables
        top_p=0.9,               # nucleus sampling
        temperature=0.7,         # menos aleatorio, más coherente
        repetition_penalty=1.2,  # evita repeticiones
        eos_token_id=core.tokenizer.eos_token_id  # fin de secuencia
    )


        tokens = core.tokenizer.convert_ids_to_tokens(outputs[0])
        tokens_string = core.decodificar_tokens(tokens)
        smiles = core.postprocesar_smiles(tokens_string)
        return smiles

    except Exception as e:
        return f"Error: {str(e)}"


# ---------- Interfaz de Gradio ----------
with gr.Blocks(title="MolGen.AI") as demo:
    gr.Markdown("## 🧬 MolGen.AI — Generación de moléculas")
    gr.Markdown("Escribí una configuración y generá una estructura SMILES basada en tu modelo.")
    inp = gr.Textbox(label="Configuración", placeholder="Ej: CCO[NH2+]...", lines=3)
    btn = gr.Button("Generar", variant="primary")
    out = gr.Textbox(label="SMILES generados", lines=6)
    btn.click(fn=run_inference, inputs=inp, outputs=out)


# ---------- Ejecución local o deploy ----------

if __name__ == "__main__":
    print("🚀 Iniciando MolGen.AI con Gradio...")
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
