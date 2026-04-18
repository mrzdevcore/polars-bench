import polars as pl
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_NEW_TOKENS = 768

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _read_schema(file_name: str, fmt: str) -> dict[str, str] | None:
    try:
        if fmt == "csv":
            schema = pl.scan_csv(file_name).collect_schema()
        else:
            schema = pl.read_parquet_schema(file_name)
        return {col: str(dtype) for col, dtype in schema.items()}
    except Exception:
        return None


def build_schema_block(tables: dict) -> str:
    lines = []
    for name, meta in tables.items():
        file_name = meta["file_name"]
        fmt = meta.get("format", "parquet")
        schema = _read_schema(file_name, fmt)
        if schema is None:
            lines.append(f"- {name}: columns unknown (file: {file_name})")
            continue
        cols = ", ".join(f"{col} ({dtype})" for col, dtype in schema.items())
        lines.append(f"- {name}: {cols}")
    return "\n".join(lines)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(payload: ChatRequest) -> ChatResponse:
    schema_block = build_schema_block(payload.tables)

    system_prompt = (
        "You write Python code using Polars to answer questions about DataFrames.\n\n"
        "The following DataFrames are ALREADY LOADED as variables with these columns. "
        "Use them directly — do NOT call pl.read_parquet or pl.read_csv:\n\n"
        f"{schema_block}\n\n"
        "Rules:\n"
        "- Use only the variable names and column names listed above.\n"
        "- Do not invent columns or rename tables.\n"
        "- Assign the final Polars DataFrame to a variable named `result`.\n"
        "- Return only the Python code. No markdown fences, no explanation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload.message},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(response=strip_code_fence(response))
