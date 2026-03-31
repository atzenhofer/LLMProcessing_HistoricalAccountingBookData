# JSONtoRDF.py
# Author: Max Vogeltanz, University of Graz, 2026

# Script for Json to RDF encoding, used here for processing entries in project "Aldersbach digital".
# in config.yaml: Can be adapted to your specific needs. Parameters can be tweaked, models and providers changed.
# Code iterates through json objects (stored in data/JSONtoRDF) and always applies same system prompt (stored in data/JSONtoRDF/prompts). Entire output then gets saved into one single json file.

# Important: Note that valid API KEYs for each provider need to be set inside .env file in the same folder as this script.
# So before running do the following: 1) make sure your API keys are defined 2) set file paths accordingly 3) set parameters, model and file paths accordingly in config.yaml


import time
import json
from pathlib import Path
from dotenv import load_dotenv
import yaml
import hashlib
from datetime import datetime, timezone

from providers import make_client


BASE_DIR = Path(__file__).resolve().parent


def load_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"⚠️ Warning: file not found: {path}. Proceeding without it.")
        return ""


def load_json_file(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def format_path(template: str, *, provider: str, model: str) -> str:
    safe_model = model.replace("/", "_").replace("\\", "_").replace(":", "_")
    return (
        template
        .replace("{PROVIDER}", provider.lower())
        .replace("{MODEL}", safe_model)
    )
    
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def write_run_log(log_path: Path, log_data: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    start_time = time.time()
    load_dotenv()

    cfg = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))

    provider = cfg["provider"]
    paths = cfg["paths"]
    gen = cfg["generation"]
    run = cfg["run"]
    cost = cfg.get("cost")
    total_cost = None

    model_name = gen["model"]
    max_tokens = int(gen.get("max_tokens", 4096))
    temperature = float(gen.get("temperature", 0))

    max_retries = int(run.get("max_retries", 3))
    base_delay = float(run.get("base_delay_seconds", 0))
    batch_size = int(run.get("batch_size", 0))  # 0 disables batch delay
    batch_delay = float(run.get("batch_delay_seconds", 0))

    input_path = (BASE_DIR / paths["input_json"]).resolve()
    system_prompt_path = (BASE_DIR / paths["system_prompt"]).resolve()

    output_rdf_path = (BASE_DIR / format_path(paths["output_rdf"], provider=provider, model=model_name)).resolve()

    system_prompt_text = load_text_file(system_prompt_path)
    system_prompt = f"{system_prompt_text}\n"

    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Input: {input_path}")
    print(f"Prompt: {system_prompt_path}")
    print(f"Output RDF: {output_rdf_path}")

    # Create provider client (same interface across providers)
    client = make_client(provider, cfg)

    # Load entries (list of dicts)
    entries = load_json_file(input_path)
    print(f"Loaded {len(entries)} entries")

    
    output_rdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite each run (change to "a" if you want append)
    with open(output_rdf_path, "w", encoding="utf-8") as rdf_file:
        total_input_tokens = 0
        total_output_tokens = 0

        for i, entry in enumerate(entries):
            entry_id = entry.get("id", "UNKNOWN_ID")
            entry_rubric = entry.get("rubric", "")
            entry_year = entry.get("year", "")
            entry_text = entry.get("entry", "")

            prompt_data = {
                "id": entry_id,
                "rubric": entry_rubric,
                "year": entry_year,
                "entry": entry_text
            }
            user_prompt = json.dumps(prompt_data, ensure_ascii=False, indent=2)

            for attempt in range(max_retries):
                try:
                    print(f"🟡 Processing entry {i+1}/{len(entries)} (Attempt {attempt+1})")

                    result = client.generate(
                        system=system_prompt,
                        user=user_prompt,
                        model=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    total_input_tokens += result.usage.input_tokens
                    total_output_tokens += result.usage.output_tokens

                    entry_output = (result.text or "").strip()

                    # Write immediately + flush
                    rdf_file.write(entry_output + "\n\n")
                    rdf_file.flush()

                    break

                except Exception as e:
                    print(f"⚠️ Error on attempt {attempt+1} for entry {i+1} (ID {entry_id}): {e}")

                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        # Write an error marker (optional)
                        rdf_file.write(f"ID: {entry_id}\nERROR: {e}\n\n")
                        rdf_file.flush()

            if base_delay > 0:
                time.sleep(base_delay)

            if batch_size > 0 and (i + 1) % batch_size == 0 and batch_delay > 0:
                print(f"⏳ Batch delay after {i+1} entries...")
                time.sleep(batch_delay)

        total_tokens = total_input_tokens + total_output_tokens
    
    print("\n--- Token Usage Summary ---")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total tokens used: {total_tokens}")

    if cost:
        total_cost = (
            (total_input_tokens / 1_000_000) * float(cost.get("input_per_1m", 0)) +
            (total_output_tokens / 1_000_000) * float(cost.get("output_per_1m", 0))
        )
        print(f"Estimated cost: ${total_cost:.4f}")
    
    end_time = time.time()
    print("\n✅ Processing complete.")
    print(f"📝 RDF version written to: {output_rdf_path}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    prompt_text_full = system_prompt_text
    prompt_hash = sha256_text(prompt_text_full)
    execution_time = end_time - start_time
    
    log_data = {
    "date_of_creation": datetime.now(timezone.utc).isoformat(),
    "provider": provider,
    "model": model_name,
    "input_file": str(input_path),
    "output_file": str(output_rdf_path),
    "system_prompt_file": str(system_prompt_path),
    "system_prompt_sha256": prompt_hash,
    "system_prompt_text": prompt_text_full,
    "run_params": {
        "max_retries": max_retries,
        "base_delay_seconds": base_delay,
        "batch_size_for_delay": batch_size,
        "batch_delay_seconds": batch_delay,
        "temperature": temperature,
        "max_tokens": max_tokens,
    },
        "results": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "average_tokens_per_entry": (total_tokens / len(entries)) if len(entries) else None,
            "estimated cost": total_cost,
            "execution_time (seconds)": execution_time,
    },
}
        
    # Put log next to output file, name includes model + timestamp
    safe_model = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_rdf_path.with_name(f"run_log_{safe_model}_{timestamp}.json")
    write_run_log(log_path, log_data)
    print(f"🧾 Run log written to: {log_path}")


if __name__ == "__main__":
    main()
