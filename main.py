import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_NEW_TOKENS = 768

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVE_K = 3
CORPUS_DIR = Path(__file__).resolve().parent / "data"

POLARS_RULES = """\
Polars syntax rules:
- Arithmetic on columns uses expressions, never strings: (pl.col("a") * pl.col("b")), NOT pl.sum("a * b").
- Use .group_by (not .groupby, which is deprecated).
- After aggregating a computed expression, always alias the result:
    .agg((pl.col("x") * pl.col("y")).sum().alias("total"))
  Without .alias(), the output column name defaults to the source column and your .sort() will miss.
- Join keys with DIFFERENT names on each side: use left_on and right_on.
    .join(orders, left_on="l_orderkey", right_on="o_orderkey")
  Only use .join(df, on="x") when the same column name "x" exists in BOTH DataFrames.
  Inspect the schema carefully — prefixed names like l_*, o_*, c_*, n_*, r_* rarely share keys.
- Chain-joins: follow the EXACT order of tables named in the question. Do not skip a step.
- Compute derived columns with .with_columns(...) before aggregating, or inline inside .agg(...).
"""

FEW_SHOT_EXAMPLES = """\
Example 1 — shared key names (use on=):
Q: Top 3 item groups by total revenue. Chain: line_items → items → item_groups.
A:
result = (
    line_items
    .join(items, on="item_id")
    .join(item_groups, on="group_id")
    .with_columns((pl.col("unit_price") * pl.col("quantity")).alias("revenue"))
    .group_by("group_name")
    .agg(pl.col("revenue").sum().round(2).alias("total_revenue"))
    .sort("total_revenue", descending=True)
    .head(3)
)

Example 2 — prefixed key names (left_on/right_on required):
Q: Order count per supplier nation, Asian suppliers only. Chain: lineitem → supplier → nation → region.
A:
result = (
    lineitem
    .join(supplier, left_on="l_suppkey",   right_on="s_suppkey")
    .join(nation,   left_on="s_nationkey", right_on="n_nationkey")
    .join(region,   left_on="n_regionkey", right_on="r_regionkey")
    .filter(pl.col("r_name") == "ASIA")
    .group_by("n_name")
    .agg(pl.col("l_orderkey").n_unique().alias("order_count"))
    .sort("order_count", descending=True)
)
"""

def _load_corpus() -> list[dict]:
    if not CORPUS_DIR.is_dir():
        return []
    examples = []
    for path in sorted(CORPUS_DIR.glob("benchmark_*.json")):
        if path.name == "benchmark_final.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for question in data.get("questions", []):
            if "question" in question and "gold_code" in question:
                examples.append(
                    {"question": question["question"], "code": question["gold_code"]}
                )
    return examples


def _build_index():
    corpus = _load_corpus()
    if not corpus or not _ST_AVAILABLE:
        return corpus, None, None
    try:
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        embeddings = embedder.encode(
            [example["question"] for example in corpus],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return corpus, embeddings, embedder
    except Exception:
        return corpus, None, None


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

CORPUS, CORPUS_EMBEDDINGS, EMBEDDER = _build_index()


def retrieve_examples(question: str, k: int = RETRIEVE_K) -> list[dict]:
    if EMBEDDER is None or CORPUS_EMBEDDINGS is None or not CORPUS:
        return []
    query_embedding = EMBEDDER.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    similarities = CORPUS_EMBEDDINGS @ query_embedding
    top_indices = np.argsort(-similarities)[:k]
    return [CORPUS[i] for i in top_indices]


def format_retrieved_examples(examples: list[dict]) -> str:
    if not examples:
        return ""
    blocks = []
    for i, example in enumerate(examples, 1):
        blocks.append(
            f"Example {i}:\nQ: {example['question']}\nA:\n{example['code']}"
        )
    return "\n\n".join(blocks)


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
    retrieved = retrieve_examples(payload.message)
    examples_block = format_retrieved_examples(retrieved) or FEW_SHOT_EXAMPLES

    system_prompt = (
        "You write Python code using Polars to answer questions about DataFrames.\n\n"
        "The following DataFrames are ALREADY LOADED as variables with these columns. "
        "Use them directly — do NOT call pl.read_parquet or pl.read_csv:\n\n"
        f"{schema_block}\n\n"
        f"{POLARS_RULES}\n"
        f"{examples_block}\n\n"
        "Output rules:\n"
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
