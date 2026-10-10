"""
TDS 2026-09 GA0 - every question that needs a URL endpoint lives in this one file.

Deployed on Render from GitHub. Start command:
    uvicorn main:app --host 0.0.0.0 --port $PORT

ENDPOINT MAP (one section per question, search for "QUESTION n"):
    GET  /                    -> lists every endpoint and which question it answers
    GET  /healthz             -> wake-up / health check
    POST /code-interpreter    -> QUESTION 5: Code Interpreter with AI Error Analysis

Naming rule: anything belonging to question N is prefixed q<N>_ / Q<N>,
except names the question text itself dictates (those are kept exactly as given).
"""

import json
import os
import re
import threading
from typing import List

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# SHARED APP SETUP (used by every question)
# =============================================================================

app = FastAPI(title="TDS GA0 endpoints")

# CORS: the grader runs in the browser at https://exam.sanand.workers.dev and
# sends a JSON POST (which triggers a preflight OPTIONS request). It sends no
# cookies, so a wildcard origin is valid here. allow_credentials must stay
# False: browsers reject "*" together with credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Pipe (course proxy for OpenAI). Set AIPIPE_TOKEN in Render -> Environment.
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_OPENAI_BASE_URL = "https://aipipe.org/openai/v1"


@app.get("/")
def index():
    """Human-friendly map of what is hosted here."""
    return {
        "service": "TDS GA0 endpoints",
        "endpoints": {
            "POST /code-interpreter": "Question 5 - Code Interpreter with AI Error Analysis",
            "GET /healthz": "Health check",
        },
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# =============================================================================
# QUESTION 5: Code Interpreter with AI Error Analysis
#   Endpoint : POST /code-interpreter
#   Request  : {"code": "<python source>"}
#   Response : {"error": [<line numbers>], "result": "<exact stdout or traceback>"}
#   Flow     : 1. execute_python_code()  2. success? return  3. AI finds error lines
# =============================================================================

Q5_AI_MODEL = "gpt-4.1-nano"
Q5_AI_TIMEOUT_SECONDS = 12

# sys.stdout is process-wide, so only one snippet may run at a time.
q5_exec_lock = threading.Lock()


class Q5CodeRequest(BaseModel):
    code: str


class Q5CodeResponse(BaseModel):
    error: List[int]  # line numbers with errors ([] when the code ran fine)
    result: str  # exact stdout, or the traceback


class ErrorAnalysis(BaseModel):
    """Structured output the AI must return (name dictated by the question)."""

    error_lines: List[int]  # Line numbers with errors


def execute_python_code(code: str) -> dict:
    """
    Q5 Part 1 - Tool function (signature and return shape dictated by the question).

    Execute Python code and return exact output.

    Returns:
        {
            "success": bool,
            "output": str  # Exact stdout or traceback
        }
    """
    import sys
    import traceback
    from io import StringIO

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Execute code.
        # One fresh dict is passed as the globals. A bare exec(code) inside a
        # function keeps globals and locals separate, which breaks any snippet
        # that defines a function and then calls it from another function
        # (e.g. recursion raises NameError).
        exec(code, {"__name__": "__main__"})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        # Get full traceback
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


def q5_error_lines_from_traceback(traceback_text: str) -> List[int]:
    """
    Q5 safety net: read the error line straight out of the traceback.

    Submitted code is compiled under the filename "<string>", so the LAST
    'File "<string>", line N' entry is the deepest line of the user's code
    that was running when it failed. Works for syntax errors too.
    """
    line_numbers = re.findall(r'File "<string>", line (\d+)', traceback_text)
    return [int(line_numbers[-1])] if line_numbers else []


def analyze_error_with_ai(code: str, traceback: str) -> List[int]:
    """
    Q5 Part 2 - AI error analysis (name and signature dictated by the question).

    Use LLM with structured output to identify error line numbers.
    Called through AI Pipe; the reply is validated with the ErrorAnalysis model.
    """
    numbered_code = "\n".join(
        f"{number}: {line}" for number, line in enumerate(code.split("\n"), start=1)
    )

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

The code was executed under the filename "<string>". Report the deepest line
of that code that was executing when the exception was raised. Ignore frames
from any other file.

CODE (each line is prefixed with its line number):
{numbered_code}

TRACEBACK:
{traceback}

Return the line number(s) where the error is located.
"""

    response = httpx.post(
        f"{AIPIPE_OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
        json={
            "model": Q5_AI_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ErrorAnalysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error_lines": {
                                "type": "array",
                                "items": {"type": "integer"},
                            }
                        },
                        "required": ["error_lines"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        timeout=Q5_AI_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    result = ErrorAnalysis.model_validate_json(
        response.json()["choices"][0]["message"]["content"]
    )
    return result.error_lines


def q5_resolve_error_lines(code: str, traceback_text: str) -> List[int]:
    """
    Q5: decide the final error lines.

    The AI is asked first (as the question requires). Its answer is then
    checked against the traceback itself, and the traceback wins whenever the
    two disagree or the AI call fails. A wrong line number fails the grader,
    so the answer must never depend on the AI being up or being right.
    """
    lines_from_traceback = q5_error_lines_from_traceback(traceback_text)

    lines_from_ai: List[int] = []
    if AIPIPE_TOKEN:
        try:
            lines_from_ai = analyze_error_with_ai(code, traceback_text)
        except Exception as ai_failure:  # network, quota, bad JSON, ...
            print(f"[Q5] AI analysis failed, using traceback: {ai_failure!r}")

    if lines_from_traceback:
        if lines_from_ai and sorted(lines_from_ai) != lines_from_traceback:
            print(
                f"[Q5] AI said {json.dumps(lines_from_ai)}, "
                f"traceback says {json.dumps(lines_from_traceback)}; using traceback"
            )
        return lines_from_traceback

    return sorted(set(lines_from_ai))


@app.post("/code-interpreter", response_model=Q5CodeResponse)
def q5_code_interpreter(request: Q5CodeRequest) -> Q5CodeResponse:
    """QUESTION 5 endpoint: run the code; on failure, report the error line(s)."""
    with q5_exec_lock:
        execution = execute_python_code(request.code)

    # Success: no AI call, output returned untouched.
    if execution["success"]:
        return Q5CodeResponse(error=[], result=execution["output"])

    # Failure: AI analysis (only now), traceback returned untouched.
    error_lines = q5_resolve_error_lines(request.code, execution["output"])
    return Q5CodeResponse(error=error_lines, result=execution["output"])


# =============================================================================
# NEXT QUESTION GOES HERE (copy the QUESTION 5 header block and rename q5_ -> qN_)
# =============================================================================
