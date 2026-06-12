#!/usr/bin/env python3
"""Simple web server for the Poodle demo config UI.

Usage:
    python server.py [--port 8080]

Endpoints:
  GET  /             → index.html
  POST /api/cost     → compute cost series from a demo_config JSON body
  POST /api/save     → save demo_config JSON to ../demo/saved_config.json
"""
import argparse
import http.server
import json
import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable so demo.* modules work
ROOT      = Path(__file__).parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from demo.demo_config import (
    DemoScenario, ModelConfig, RequestConfig,
    TokenConfig, ModelDevConfig, ValidationConfig,
    Model, TaskDetectionMethod,
)
import cost_break_even.price_estimation as pe
from cost_break_even.price_estimation import (
    single_model_price, poodle_price, MODEL_PRICING_PER_1M,
    INPUT, OUTPUT, analytical_break_even,
)

SCENARIOS_PATH     = ROOT / "scenarios.json"
CUSTOM_MODELS_PATH   = ROOT  / "custom_models.json"
MEASURED_RESULTS_CSV = REPO_ROOT / "demo" / "measured-results.csv"

# ── Scenario persistence ───────────────────────────────────────────────────

def _load_scenarios() -> dict:
    if SCENARIOS_PATH.exists():
        return json.loads(SCENARIOS_PATH.read_text())
    return {}

def _save_scenarios(data: dict):
    SCENARIOS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Custom model persistence ───────────────────────────────────────────────
# Each entry: { "value": str, "name": str, "provider": str, "input": float, "output": float }
_CUSTOM_META: list[dict] = []

def _load_custom_models():
    if CUSTOM_MODELS_PATH.exists():
        data = json.loads(CUSTOM_MODELS_PATH.read_text())
        for m in data:
            _CUSTOM_META.append(m)
            pe.CUSTOM_MODEL_PRICING[m["value"]] = {INPUT: m["input"], OUTPUT: m["output"]}

def _save_custom_models():
    CUSTOM_MODELS_PATH.write_text(json.dumps(_CUSTOM_META, indent=2))

_load_custom_models()


# ── Model display metadata ─────────────────────────────────────────────────

_PRETTY_NAMES = {
    Model.BERT_80M:         "BERT 80M",
    Model.LLAMA_8B:         "Llama 3.1 8B",
    Model.LLAMA_70B_TURBO:  "Llama 3.3 70B Turbo",
    Model.LLAMA_405B_TURBO: "Llama 3.1 405B Turbo",
    Model.GPT_4_1:          "GPT-4.1",
    Model.GPT_4_1_MINI:     "GPT-4.1 Mini",
    Model.GPT_4_1_NANO:     "GPT-4.1 Nano",
}

_PROVIDERS = {
    Model.BERT_80M:         "Together AI",
    Model.LLAMA_8B:         "Together AI",
    Model.LLAMA_70B_TURBO:  "Together AI",
    Model.LLAMA_405B_TURBO: "Together AI",
    Model.GPT_4_1:          "OpenAI",
    Model.GPT_4_1_MINI:     "OpenAI",
    Model.GPT_4_1_NANO:     "OpenAI",
}

def _build_model_table() -> list[dict]:
    rows = []
    for model, pricing in MODEL_PRICING_PER_1M.items():
        rows.append({
            "value":    model.value,
            "name":     _PRETTY_NAMES.get(model, model.value),
            "provider": _PROVIDERS.get(model, "—"),
            "input":    pricing[INPUT],
            "output":   pricing[OUTPUT],
            "custom":   False,
        })
    for m in _CUSTOM_META:
        rows.append({**m, "custom": True})
    return rows


# ── Config deserialization ─────────────────────────────────────────────────

def _parse_model(value: str):
    """Return a Model enum if known, or the raw string for custom models."""
    try:
        return Model(value)
    except ValueError:
        return value  # custom model — stored as plain string


def scenario_from_dict(data: dict) -> DemoScenario:
    """Build a DemoScenario from a plain dict (as sent by the browser)."""
    m = data.get("models", {})
    t = data.get("tokens", {})
    return DemoScenario(
        models=ModelConfig(
            large_model=_parse_model(m.get("large_model", "gpt-4.1")),
            small_model=_parse_model(m.get("small_model", "bert-80M")),
        ),
        requests=RequestConfig(**data.get("requests", {})),
        tokens=TokenConfig(
            input=t.get("input", ""),
            prompt=t.get("prompt", ""),
            task_detection_method=TaskDetectionMethod(
                t.get("task_detection_method", "wrapper_prompt")
            ),
            wrapper_prompt=t.get("wrapper_prompt", ""),
            wrapped_requests_percent=t.get("wrapped_requests_percent", 1.0),
            output=t.get("output", ""),
            wrapped_output=t.get("wrapped_output", ""),
        ),
        dev=ModelDevConfig(**data.get("dev", {})),
        validation=ValidationConfig(**data.get("validation", {})),
    )


def _load_measured_results() -> list[dict]:
    """Parse demo/measured-results.csv into a list of row dicts."""
    import csv
    if not MEASURED_RESULTS_CSV.exists():
        return []
    rows = []
    with open(MEASURED_RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            # Coerce numeric fields; keep "None" as null
            for key in ("switch_req", "dev_time"):
                try:
                    row[key] = int(row[key])
                except (ValueError, KeyError):
                    row[key] = None
            for key in ("accuracy", "items-per-second", "throughput-1m"):
                try:
                    row[key] = float(row[key])
                except (ValueError, KeyError):
                    row[key] = None
            rows.append(row)
    return rows


# ── HTTP handler ───────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}]  {fmt % args}")

    # ── GET ────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(ROOT / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/example":
            scenario = DemoScenario.get_example_scenario()
            self._send(200, "application/json",
                       json.dumps(scenario.to_dict(), indent=2,
                                  default=lambda o: o.value if hasattr(o, 'value') else str(o)).encode())
        elif self.path == "/api/models":
            self._send(200, "application/json",
                       json.dumps(_build_model_table()).encode())
        elif self.path == "/api/scenarios":
            scenarios = _load_scenarios()
            # Return as list of {key, config} so order is preserved
            payload = [{"key": k, "config": v} for k, v in scenarios.items()]
            self._send(200, "application/json", json.dumps(payload).encode())
        elif self.path == "/api/use-cases":
            uc_path = ROOT / "use_cases.json"
            data = json.loads(uc_path.read_text()) if uc_path.exists() else {}
            self._send(200, "application/json", json.dumps(data).encode())
        elif self.path == "/api/measured-results":
            rows = _load_measured_results()
            self._send(200, "application/json", json.dumps(rows).encode())
        elif self.path == "/api/baseline-single":
            bl_path = ROOT / "baseline.json"
            data = json.loads(bl_path.read_text()) if bl_path.exists() else {"models": []}
            self._send(200, "application/json", json.dumps(data).encode())
        elif self.path == "/api/baseline":
            bl_path = ROOT / "baseline_search.json"
            data = json.loads(bl_path.read_text()) if bl_path.exists() else {"models": []}
            self._send(200, "application/json", json.dumps(data).encode())
        elif self.path == "/api/proxy-scoring":
            ps_path = ROOT / "proxy_scoring_500.json"
            data = json.loads(ps_path.read_text()) if ps_path.exists() else {"models": []}
            self._send(200, "application/json", json.dumps(data).encode())
        elif self.path == "/api/proxy-ft-small":
            ft_path = ROOT / "ft_small_model-3.json"
            data = json.loads(ft_path.read_text()) if ft_path.exists() else {}
            self._send(200, "application/json", json.dumps(data).encode())
        elif self.path == "/api/proxy-scoring-sh":
            ps_path = ROOT / "proxy_scoring_sh.json"
            data = json.loads(ps_path.read_text()) if ps_path.exists() else {"rounds": []}
            self._send(200, "application/json", json.dumps(data).encode())
        elif self.path == "/api/model-mapping":
            mm_path = ROOT / "models.json"
            data = json.loads(mm_path.read_text()) if mm_path.exists() else {}
            self._send(200, "application/json", json.dumps(data).encode())
        elif self.path == "/api/switch-time-llm":
            # TODO: implement — return null to show ??? in UI
            self._send(200, "application/json", json.dumps({"value": None}).encode())
        elif self.path == "/api/switch-time-poodle":
            # TODO: implement — return null to show ??? in UI
            self._send(200, "application/json", json.dumps({"value": None}).encode())
        else:
            self._send(404, "text/plain", b"Not found")

    # ── POST ───────────────────────────────────────────────────────────

    def do_POST(self):
        body = self._read_body()
        if body is None:
            return

        if self.path == "/api/cost":
            self._handle_cost(body)
        elif self.path == "/api/scenarios/save":
            self._handle_scenario_save(body)
        elif self.path == "/api/scenarios/delete":
            self._handle_scenario_delete(body)
        elif self.path == "/api/models":
            self._handle_add_model(body)
        elif self.path == "/api/models/delete":
            self._handle_delete_model(body)
        else:
            self._send(404, "application/json",
                       json.dumps({"error": "not found"}).encode())

    # ── Route handlers ─────────────────────────────────────────────────

    def _handle_cost(self, data: dict):
        try:
            scenario = scenario_from_dict(data)
            # 100 linearly spaced points from 1 to expected_requests
            n = 100
            max_r = scenario.requests.expected_requests
            request_values = [max(1, int(max_r * i / (n - 1))) for i in range(n)]
            from dataclasses import replace as dc_replace
            reqs, base_list, poodle_list, savings_list = [], [], [], []
            for r in request_values:
                s = dc_replace(scenario, requests=dc_replace(scenario.requests, expected_requests=r))
                b = single_model_price(s.models.large_model, s)
                p = poodle_price(s)
                reqs.append(r)
                base_list.append(b)
                poodle_list.append(p)
                savings_list.append(b - p)

            result = {
                "requests":       reqs,
                "base_prices":    base_list,
                "poodle_prices":  poodle_list,
                "poodle_savings": savings_list,
                "break_even":     analytical_break_even(scenario),
            }
            self._send(200, "application/json", json.dumps(result).encode())
        except Exception as e:
            self._send(500, "application/json",
                       json.dumps({"error": str(e)}).encode())

    def _handle_add_model(self, data: dict):
        try:
            name     = str(data["name"]).strip()
            provider = str(data.get("provider", "Custom")).strip() or "Custom"
            inp      = float(data["input"])
            out      = float(data["output"])
            if not name:
                raise ValueError("name is required")
            # Use a slug as the value key
            value = name.lower().replace(" ", "-")
            # Remove existing entry with same value if present
            _CUSTOM_META[:] = [m for m in _CUSTOM_META if m["value"] != value]
            entry = {"value": value, "name": name, "provider": provider,
                     "input": inp, "output": out}
            _CUSTOM_META.append(entry)
            pe.CUSTOM_MODEL_PRICING[value] = {INPUT: inp, OUTPUT: out}
            _save_custom_models()
            self._send(200, "application/json",
                       json.dumps(_build_model_table()).encode())
        except (KeyError, ValueError) as e:
            self._send(400, "application/json",
                       json.dumps({"error": str(e)}).encode())

    def _handle_delete_model(self, data: dict):
        value = data.get("value", "")
        _CUSTOM_META[:] = [m for m in _CUSTOM_META if m["value"] != value]
        pe.CUSTOM_MODEL_PRICING.pop(value, None)
        _save_custom_models()
        self._send(200, "application/json",
                   json.dumps(_build_model_table()).encode())

    def _handle_scenario_save(self, data: dict):
        name   = str(data.get("name", "")).strip()
        config = data.get("config")
        if not name:
            self._send(400, "application/json",
                       json.dumps({"error": "name is required"}).encode())
            return
        if not isinstance(config, dict):
            self._send(400, "application/json",
                       json.dumps({"error": "config is required"}).encode())
            return
        scenarios = _load_scenarios()
        scenarios[name] = config
        _save_scenarios(scenarios)
        print(f"  Scenario '{name}' saved → scenarios.json")
        payload = [{"key": k, "config": v} for k, v in scenarios.items()]
        self._send(200, "application/json", json.dumps(payload).encode())

    def _handle_scenario_delete(self, data: dict):
        name = str(data.get("name", "")).strip()
        scenarios = _load_scenarios()
        scenarios.pop(name, None)
        _save_scenarios(scenarios)
        payload = [{"key": k, "config": v} for k, v in scenarios.items()]
        self._send(200, "application/json", json.dumps(payload).encode())

    # ── Helpers ────────────────────────────────────────────────────────

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            self._send(400, "application/json",
                       json.dumps({"error": str(e)}).encode())
            return None

    def _serve_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send(404, "text/plain", b"File not found")
            return
        self._send(200, content_type, data)

    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Poodle config UI server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="localhost",
                        help="Bind address (use 0.0.0.0 inside Docker)")
    args = parser.parse_args()

    server = http.server.HTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  🐩  Poodle Config UI")
    print(f"  Running at  {url}")
    print(f"  Scenarios   {SCENARIOS_PATH.relative_to(ROOT)}")
    print(f"\n  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
