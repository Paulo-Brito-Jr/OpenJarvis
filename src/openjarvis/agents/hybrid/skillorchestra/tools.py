"""The three SkillOrchestra tools: search, enhance_reasoning (code), answer.

Faithful port of ``orchestration/eval_frames.py:call_tool`` — same worker
prompts and extraction. Security deltas forced by the OpenJarvis environment:

* ``search`` — the original POSTs to a FAISS wiki retriever service. We
  route configured retrievers through the SSRF-safe HTTP transport; with no
  retriever configured we fall back to provider-side web search.
* ``enhance_reasoning`` — generated Python is retained for inspection but is
  never executed on the host. A verified isolated sandbox adapter is required
  before execution can be re-enabled.
* in-tool correctness check — the original ``answer`` tool LLM-judges the
  prediction against the gold answer inside ``call_tool``. OpenJarvis
  scores with its own harness judge downstream, so we only return the
  prediction; no gold answer is threaded in.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openjarvis.core.cancellation import AgentCancelledError, raise_if_cancelled
from openjarvis.tools.http_request import HttpRequestTool

from .._base import (
    GEMINI_SEARCH_COST_PER_CALL,
    OPENAI_WEB_SEARCH_COST_PER_CALL,
    WEB_SEARCH_COST_PER_CALL,
    build_web_search_tool,
    tavily_search_context,
)
from .pool import ModelSpec, call_alias

# Cloud endpoints with a server-side web-search agent loop wired in
# `_base.py`. Anything else (openrouter, vllm, unknown) can't ground.
_SEARCH_CAPABLE_ENDPOINTS = ("anthropic", "openai", "gemini")

# ---------------------------------------------------------------------------
# Tool schemas — orchestration/tools.json, in Anthropic + OpenAI shapes.
# ---------------------------------------------------------------------------

_SEARCH_DESC = "Search for missing information."
_CODE_DESC = (
    "Write and execute Python code to compute intermediate results for the problem."
)
_ANSWER_DESC = (
    "Extract the final answer when you have gathered enough information "
    "to answer the problem."
)

_ENUMS = {
    "search": ["search-1", "search-2", "search-3"],
    "enhance_reasoning": ["reasoner-1", "reasoner-2", "reasoner-3"],
    "answer": [
        "answer-1",
        "answer-2",
        "answer-3",
        "answer-4",
        "answer-math-1",
        "answer-math-2",
    ],
}


def _model_prop(tool: str) -> Dict[str, Any]:
    return {
        "type": "string",
        "description": (
            f"Model alias for the {tool} tool. Choose one of: "
            + ", ".join(_ENUMS[tool])
        ),
        "enum": _ENUMS[tool],
    }


def anthropic_tools() -> List[Dict[str, Any]]:
    """The 3 orchestrator tools in Anthropic ``input_schema`` shape."""
    out = []
    for name, desc in (
        ("search", _SEARCH_DESC),
        ("enhance_reasoning", _CODE_DESC),
        ("answer", _ANSWER_DESC),
    ):
        out.append(
            {
                "name": name,
                "description": desc,
                "input_schema": {
                    "type": "object",
                    "properties": {"model": _model_prop(name)},
                    "required": ["model"],
                },
            }
        )
    return out


def openai_tools() -> List[Dict[str, Any]]:
    """The 3 orchestrator tools in OpenAI ``function`` shape."""
    out = []
    for name, desc in (
        ("search", _SEARCH_DESC),
        ("enhance_reasoning", _CODE_DESC),
        ("answer", _ANSWER_DESC),
    ):
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {"model": _model_prop(name)},
                        "required": ["model"],
                    },
                },
            }
        )
    return out


def gemini_tools() -> List[Dict[str, Any]]:
    """The 3 orchestrator tools in Gemini function-declaration shape."""
    out = []
    for name, desc in (
        ("search", _SEARCH_DESC),
        ("enhance_reasoning", _CODE_DESC),
        ("answer", _ANSWER_DESC),
    ):
        out.append(
            {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {"model": _model_prop(name)},
                    "required": ["model"],
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# enhance_reasoning / code  —  eval_frames.py:659-812
# ---------------------------------------------------------------------------


def run_code(
    agent: Any,
    spec: ModelSpec,
    *,
    context_str: str,
    problem: str,
    bash_timeout_s: int = 60,
) -> Dict[str, Any]:
    """Generate self-contained Python without executing it on the host.

    The historical implementation passed model-authored source directly to
    the host Python interpreter. Capability checks cannot make that an
    isolation boundary, so execution remains disabled until a verified
    sandbox adapter is wired.
    """
    raise_if_cancelled()
    del bash_timeout_s
    agent._require_action(
        "code:skillorchestra",
        ["code:execute"],
        tool_name="skillorchestra_code",
    )
    prompt = (
        context_str.strip()
        + "\n\n"
        + f"Question: {problem}\nInstead of directly answering the question, "
        "please write additional python code that will give intermidiate "
        "results after execution. Wrap the code within ```python and ```. "
        "The code should be self-contained with all the import and "
        "initialization."
    )
    text, p, c, cost = call_alias(
        agent,
        spec,
        user=prompt,
        max_tokens=8000,
        temperature=1.0,
    )
    generated_code = ""
    if "```python" in text:
        generated_code = text.split("```python")[-1].split("```")[0]

    exec_result = ""
    security_disabled = bool(generated_code.strip())
    if security_disabled:
        exec_result = (
            "[execution disabled: verified isolated sandbox executor required]"
        )
    return {
        "tool": "enhance_reasoning",
        "model": spec.model,
        "alias": spec.alias,
        "generated_code": generated_code,
        "exec_result": exec_result,
        "response": text,
        "tokens_in": p,
        "tokens_out": c,
        "cost_usd": cost,
        "is_local": spec.is_local,
        "security_disabled": security_disabled,
    }


# ---------------------------------------------------------------------------
# answer  —  eval_frames.py:814-997
# ---------------------------------------------------------------------------


def run_answer(
    agent: Any,
    spec: ModelSpec,
    *,
    context_str: str,
    problem: str,
    max_tokens: int = 40000,
) -> Dict[str, Any]:
    """Generate the final answer with ``spec`` and extract the prediction.

    The original branches the prompt by model family: Qwen-3 / Qwen-math
    get a ``\\boxed{}`` system prompt; GPT-5 / Claude (and we extend this
    to every other model) get the ``<think>/<answer>`` instruction. The
    in-tool LLM correctness check is dropped — OpenJarvis scores
    downstream.
    """
    raise_if_cancelled()
    base = context_str.strip() + "\n\n" + problem
    model_l = spec.model.lower()
    system: Optional[str] = None
    boxed = False

    if "qwen3" in model_l and "235" not in model_l:
        system = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        user = base
        boxed = True
    elif "qwen2.5-math" in model_l or "qwen-2.5-math" in model_l:
        system = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        user = base
        boxed = True
    else:
        user = base + (
            "\n\nTake a deep breath and think hard with high reasoning, wrap "
            "the thoughts within <think> and </think>, and wrap only the "
            "exact answer without any explanation within <answer> and "
            "</answer>.Output using the following format:\n<think>\n...\n"
            "</think>\n<answer>\n...\n</answer>"
        )

    text, p, c, cost = call_alias(
        agent,
        spec,
        user=user,
        system=system,
        max_tokens=max_tokens,
        temperature=1.0,
    )

    pred = ""
    if boxed and "\\boxed{" in text:
        pred = "}".join(text.split("\\boxed{")[-1].split("}")[:-1]).strip()
    elif "<answer>" in text:
        pred = text.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        pred = text.strip()
    # Original: a >500-word "answer" is treated as a non-answer.
    if len(pred.split()) > 500:
        pred = ""

    return {
        "tool": "answer",
        "model": spec.model,
        "alias": spec.alias,
        "pred": pred,
        "response": text,
        "tokens_in": p,
        "tokens_out": c,
        "cost_usd": cost,
        "is_local": spec.is_local,
    }


# ---------------------------------------------------------------------------
# search  —  eval_frames.py:999-1096
# ---------------------------------------------------------------------------


def run_search(
    agent: Any,
    spec: ModelSpec,
    *,
    context_str: str,
    problem: str,
    retriever_url: Optional[str] = None,
    topk: int = 150,
    web_search_max_uses: int = 5,
    search_backend: str = "provider",
    tavily_max_results: int = 5,
) -> Dict[str, Any]:
    """Write a search query with ``spec``, then retrieve documents.

    Query generation is the original verbatim worker prompt. Retrieval:
    if ``retriever_url`` is set we POST the original ``/retrieve`` payload;
    otherwise we fall back to Anthropic ``web_search`` (the documented
    OpenJarvis substitution for the missing FAISS wiki index).
    """
    raise_if_cancelled()
    prompt = (
        context_str.strip()
        + "\n\n"
        + f"Question: {problem}\nInstead of directly answering the question, "
        "please think hard and write a concise query to search Wikipedia. "
        "Wrap the query within <query> and </query>."
    )
    text, p, c, cost = call_alias(
        agent,
        spec,
        user=prompt,
        max_tokens=8000,
        temperature=1.0,
    )
    # Query generation is a separate inference boundary.  Do not start a
    # search after its caller has disconnected.
    raise_if_cancelled()
    if "<query>" in text:
        query = text.split("<query>")[-1].split("</query>")[0].strip()
    else:
        query = ""
    if len(query) < 10:
        query = problem

    contents: List[str] = []
    search_uses = 0
    if search_backend == "tavily":
        res = tavily_search_context(
            query,
            max_results=tavily_max_results,
            action_authorizer=agent._action_authorizer,
        )
        contents.append(res["text"])
        search_uses = int(res["n_searches"])
        cost += float(res["cost_usd"])
    elif retriever_url:
        # Route retriever POSTs through the hardened HTTP transport so DNS
        # pinning and every redirect receive an SSRF check. Scan model-authored
        # query text before it crosses the boundary.
        endpoint = f"{retriever_url.rstrip('/')}/retrieve"
        agent._require_action(
            endpoint,
            ["network:fetch", "network:mutate"],
            tool_name="skillorchestra_retriever",
        )
        confirmation_denied = agent._action_authorizer.confirm_action(
            "skillorchestra_retriever",
            {"url": endpoint, "method": "POST"},
        )
        if confirmation_denied is not None:
            raise PermissionError(confirmation_denied.content)
        guarded_query = agent._action_authorizer.guard_outbound_content(
            query[:390],
            endpoint,
        )
        raise_if_cancelled()
        payload = {
            "queries": [guarded_query],
            "topk": topk,
            "return_scores": True,
        }
        try:
            raise_if_cancelled()
            response = HttpRequestTool().execute(
                url=endpoint,
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps(payload),
                timeout=60,
            )
            if not response.success:
                raise RuntimeError(response.content)
            status_code = int(response.metadata.get("status_code", 0))
            if not 200 <= status_code < 300:
                raise RuntimeError(f"retriever returned HTTP status {status_code}")
            results = json.loads(response.content)
            for r in results[0]:
                doc = r.get("document", {})
                if "content" in doc:
                    contents.append(doc["content"])
                elif "contents" in doc:
                    contents.append(doc["contents"])
            agent.record_trace_event(
                {
                    "kind": "skillorchestra_retriever",
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "success": bool(response.success),
                    "result_count": len(contents),
                }
            )
            # The HTTP result is now represented in the trace and accumulated
            # context; only then surface cancellation from an in-flight call.
            raise_if_cancelled()
        except AgentCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            contents.append(f"[retriever error: {exc}]")
    else:
        # Substitution path — server-side web search via the cloud's
        # `_base` agent loop. The search-capable helpers all talk to
        # ``agent._cloud_model`` with their provider SDK. If this cell
        # routes the cloud through an endpoint with no search wiring
        # (openrouter / vllm), the search stage would produce nothing and
        # the orchestrator answers the GAIA question blind — fail loud.
        endpoint = agent._cloud_endpoint
        if endpoint not in _SEARCH_CAPABLE_ENDPOINTS:
            raise ValueError(
                f"skillorchestra search fell back to web_search but "
                f"cloud_endpoint={endpoint!r}; server-side web_search is "
                "wired for anthropic / openai / gemini executors only. "
                "Set method_cfg.retriever_url to a FAISS retriever, route "
                "this cell's cloud through one of those endpoints, or "
                "override the search-* aliases in method_cfg.model_pool — "
                "otherwise the search stage produces nothing and the "
                "orchestrator answers blind."
            )
        search_user = f"Search the web and report findings for: {query}"
        try:
            if endpoint == "anthropic":
                ws_text, wp, wc, n_searches, _ = agent._call_anthropic_agent(
                    agent._cloud_model,
                    user=search_user,
                    max_tokens=4096,
                    temperature=1.0,
                    tools=[build_web_search_tool(web_search_max_uses)],
                    max_turns=4,
                    action_authorizer=agent._action_authorizer,
                )
                ws_cost_per_call = WEB_SEARCH_COST_PER_CALL
            elif endpoint == "openai":
                ws_text, wp, wc, n_searches, _ = agent._call_openai_agent(
                    agent._cloud_model,
                    user=search_user,
                    max_tokens=4096,
                    temperature=1.0,
                    max_turns=4,
                    action_authorizer=agent._action_authorizer,
                )
                ws_cost_per_call = OPENAI_WEB_SEARCH_COST_PER_CALL
            else:  # gemini
                ws_text, wp, wc, n_searches, _ = agent._call_gemini_agent(
                    agent._cloud_model,
                    user=search_user,
                    max_tokens=4096,
                    temperature=1.0,
                    max_turns=4,
                    action_authorizer=agent._action_authorizer,
                )
                ws_cost_per_call = GEMINI_SEARCH_COST_PER_CALL
            contents.append(ws_text)
            p += wp
            c += wc
            search_uses = n_searches
            cost += agent.cost_usd(agent._cloud_model, wp, wc)
            cost += n_searches * ws_cost_per_call
        except (AgentCancelledError, PermissionError):
            raise
        except Exception as exc:  # noqa: BLE001
            contents.append(f"[web_search error: {exc}]")

    return {
        "tool": "search",
        "model": spec.model,
        "alias": spec.alias,
        "query": query,
        "search_results_data": contents,
        "tokens_in": p,
        "tokens_out": c,
        "cost_usd": cost,
        "web_search_uses": search_uses,
        "is_local": spec.is_local,
    }
