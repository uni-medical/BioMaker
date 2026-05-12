import json
import logging
import re
from threading import BoundedSemaphore, local
from typing import Any, Callable

from ..backend import FunctionSpec, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..utils.config import Config
from ..utils.events import append_event
from ..utils.response import truncate_text, wrap_code
from .base_agent import BaseAgent, RoundTask
from .context_builder import ContextBuilder, prompt_shape

logger = logging.getLogger("biomaker")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

MONAI_QUERY_HINTS = {
    "monai",
    "imagedataset",
    "arraydataset",
    "cachedataset",
    "persistentdataset",
    "loadimage",
    "loadimaged",
    "ensurechannelfirst",
    "ensurechannelfirstd",
    "scaleintensity",
    "scaleintensityd",
    "resized",
    "randflipd",
    "randrotate90d",
    "dynunet",
    "diceloss",
    "diceceloss",
}

TOOL_OBSERVATION_TOTAL_CHARS = 12000
TOOL_OBSERVATION_ENTRY_CHARS = 1600
NODE_TOOL_CONCURRENCY_LIMIT = 4
_NODE_TOOL_SEMAPHORE = BoundedSemaphore(NODE_TOOL_CONCURRENCY_LIMIT)
LOW_SIGNAL_API_SYMBOLS = {
    "accuracy",
    "auc",
    "batch",
    "check",
    "class",
    "code",
    "compose",
    "configuration",
    "data",
    "dataloader",
    "dataset",
    "dermdataset",
    "false",
    "file",
    "filenotfounderror",
    "image",
    "label",
    "memory",
    "metric",
    "none",
    "path",
    "prediction",
    "preparation",
    "result",
    "runtimeerror",
    "score",
    "target",
    "traceback",
    "true",
    "typeerror",
    "valueerror",
}
ERROR_EVIDENCE_TERMS = {
    "attributeerror",
    "exception",
    "importerror",
    "modulenotfounderror",
    "runtimeerror",
    "signature",
    "traceback",
    "typeerror",
    "unexpected keyword",
    "valueerror",
}

tool_request_func_spec = FunctionSpec(
    name="submit_tool_requests",
    json_schema={
        "type": "object",
        "properties": {
            "requests": {
                "type": "array",
                "description": "Optional information requests before final code generation. Return [] when the prompt already contains enough information.",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": ["inspect_api", "search_docs", "web_search"],
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason this information is needed.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query for search_docs or web_search. Do not include local sample values, file paths, experiment metrics, or private task text.",
                        },
                        "code": {
                            "type": "string",
                            "description": "Short read-only Python snippet for inspect_api.",
                        },
                    },
                    "required": ["tool", "reason"],
                },
            }
        },
        "required": ["requests"],
    },
    description="Request optional tool observations before writing the final candidate code.",
)

draft_plan_batch_func_spec = FunctionSpec(
    name="create_draft_plan_batch",
    json_schema={
        "type": "object",
        "properties": {
            "plans": {
                "type": "array",
                "description": "Diverse textual root-draft plans.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short distinct strategy name.",
                        },
                        "plan": {
                            "type": "string",
                            "description": "Concrete 3-5 sentence textual plan for one root draft. Do not include code.",
                        },
                    },
                    "required": ["name", "plan"],
                },
            }
        },
        "required": ["plans"],
    },
    description="Create a batch of textual root-draft plans before code generation.",
)


class ProposalAgent(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)
        self.profile = "You are a pragmatic biomedical machine learning engineer. "
        self.knowledge_base = None
        self.context_builder = ContextBuilder(self)
        self._generation_context = local()

    def set_pending_round_nodes(self, nodes: list[Node]) -> None:
        """Expose generated-but-unevaluated same-round proposals to later prompts."""
        self.set_generation_context(pending_round_nodes=nodes)

    def clear_pending_round_nodes(self) -> None:
        self.clear_generation_context()

    def set_generation_context(
        self,
        *,
        pending_round_nodes: list[Node],
        round_item_index: int | None = None,
        round_item_count: int | None = None,
        scheduled_step: int | None = None,
        draft_plan: str | None = None,
    ) -> None:
        """Set generation-local prompt context for the current worker thread."""
        self._generation_context.value = {
            "pending_round_nodes": list(pending_round_nodes),
            "round_item_index": round_item_index,
            "round_item_count": round_item_count,
            "scheduled_step": scheduled_step,
            "draft_plan": draft_plan,
        }

    def clear_generation_context(self) -> None:
        self._generation_context.value = {}

    def generation_context(self) -> dict[str, Any]:
        return getattr(self._generation_context, "value", {}) or {}

    def pending_round_nodes_for(self, selected_parent: Node | None) -> list[Node]:
        """Return same-round generated siblings that have not executed yet."""
        pending_round_nodes = self.generation_context().get("pending_round_nodes", [])
        nodes = []
        for node in pending_round_nodes:
            if node.tag == "generating" and not str(node.code or "").strip():
                continue
            same_parent = node.parent is selected_parent
            same_virtual_root = node.parent is None and selected_parent is None
            if same_parent or same_virtual_root:
                nodes.append(node)
        return nodes

    def _is_monai_query(self, query: str) -> bool:
        normalized = query.lower().replace(".", "").replace("_", "")
        return any(hint in normalized for hint in MONAI_QUERY_HINTS)

    def _extract_api_symbols(self, text: str, limit: int = 12) -> list[str]:
        """Extract high-signal API/error identifiers for retrieval query construction."""
        patterns = [
            r"\b(?:monai|torch|torchvision|numpy|pandas|sklearn)(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
            r"\b__[A-Za-z0-9_]+__\b",
            r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b",
            r"\b[A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*\b",
        ]
        symbols: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, text):
                normalized = re.sub(r"[^a-z0-9]+", "", match.lower())
                if (
                    len(normalized) < 4
                    or normalized in LOW_SIGNAL_API_SYMBOLS
                    or normalized in seen
                ):
                    continue
                seen.add(normalized)
                symbols.append(match)
                if len(symbols) >= limit:
                    return symbols
        return symbols

    def _has_exact_api_reference(self, query: str) -> bool:
        return bool(
            re.search(
                r"\b(?:monai|torch|torchvision|numpy|pandas|sklearn)(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
                query,
            )
            or re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b", query)
            or re.search(r"\b__[A-Za-z0-9_]+__\b", query)
        )

    def _is_monai_evidence_query(self, query: str) -> bool:
        lowered = query.lower()
        if not self._is_monai_query(query):
            return False
        if any(term in lowered for term in ERROR_EVIDENCE_TERMS):
            return True
        if self._has_exact_api_reference(query):
            return True
        return False

    def _is_local_contract_query(self, query: str) -> bool:
        lowered = query.lower()
        return (
            "evaluator" in lowered
            or "score_val" in lowered
            or "prediction_spec" in lowered
        )

    def _augment_doc_query(self, query: str, prompt: dict[str, Any] | None) -> str:
        if prompt is None:
            return query
        context_parts = []
        for key in [
            "Tree Records",
            "Previous (buggy) implementation",
            "Previous solution",
            "Best known solution",
        ]:
            if key in prompt:
                context_parts.append(str(prompt[key]))
        if "Execution output" in prompt:
            relevant_lines = []
            for line in str(prompt["Execution output"]).splitlines():
                lowered = line.lower()
                if (
                    "monai" in lowered
                    or "metatensor" in lowered
                    or "runtimeerror" in lowered
                    or "typeerror" in lowered
                ):
                    relevant_lines.append(line)
            context_parts.append("\n".join(relevant_lines))
        symbols = self._extract_api_symbols("\n".join(context_parts))
        missing = [symbol for symbol in symbols if symbol.lower() not in query.lower()]
        if not missing:
            return query
        return f"{query}\nRelevant exact identifiers from current context: {' '.join(missing)}"

    def _wrap_code_context(self, code: str, max_chars: int) -> str:
        return wrap_code(
            truncate_text(code, max_chars=max_chars, marker="code trimmed")
        )

    def query_knowledge_base(self, query: str, topk: int = 3) -> str:
        result = ""
        if self.acfg.enable_monai_knowledge_base:
            if not self._is_monai_query(query):
                logger.debug("Skipping MONAI RAG for non-MONAI query: %s", query)
                return ""
            logger.debug("Querying MONAI knowledge base with query: %s", query)
            try:
                if self.knowledge_base is None:
                    from ..monai_rag.query_rag_db import MED_KNOWLEDGE_RAG

                    self.knowledge_base = MED_KNOWLEDGE_RAG(
                        model=self.acfg.knowledge_base.model
                    )
                result = self.knowledge_base.search(
                    query, k=topk, enable_query2doc=self.acfg.enable_query2doc
                )
            except Exception as e:
                logger.warning(f"Failed to query knowledge base with error: {e}.")
            logger.debug(f"Querying knowledge base get result:\n {result[:200]}")
        return result

    def query_web(self, query: str, topk: int | None = None) -> str:
        if not self.acfg.enable_web_search:
            return ""

        query = self._sanitize_web_query(query)
        if not query:
            return ""

        try:
            from ..tools import web_search

            result = web_search(
                query,
                topk=topk or self.acfg.web_search_topk,
            )
        except Exception as e:
            logger.warning("Failed to search the web with error: %s.", e)
            return ""
        if not result:
            logger.debug("Web search returned no content for query: %s", query)
            return ""
        logger.debug("Web search result preview:\n%s", result[:200])
        return result

    def _sanitize_web_query(self, query: str) -> str:
        """Keep web queries focused on public API/error terms."""
        lines = []
        for raw_line in str(query).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if any(
                marker in lowered
                for marker in [
                    "/home/",
                    "./input",
                    "./working",
                    "biomaker_splits",
                    "agent_train",
                    "score_val",
                    "agent_test",
                    "validation metric",
                    "private_gt",
                ]
            ):
                continue
            lines.append(line)
        cleaned = " ".join(" ".join(lines).split())
        symbols = self._extract_api_symbols(cleaned, limit=16)
        if symbols:
            return " ".join(symbols)
        return cleaned[:200]

    def _tool_decision_context(
        self, stage: str, prompt: dict[str, Any]
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "Stage": stage,
            "Task Description": truncate_text(str(self.task_desc), max_chars=1200),
        }
        summary = self.get_data_summary()
        if summary:
            context["Data and evaluation summary"] = truncate_text(
                summary, max_chars=1600
            )
        context["Cached environment facts"] = truncate_text(
            self.environment_facts, max_chars=1600
        )
        if "Current Selection" in prompt:
            context["Current Selection"] = prompt["Current Selection"]
        if "Tree Records" in prompt:
            context["Tree Records"] = prompt["Tree Records"]
        for key in [
            "Memory",
            "Selected parent experiment facts",
            "Previous solution",
            "Best known solution",
            "Execution output",
            "Tool Observations",
        ]:
            if key in prompt:
                context[key] = truncate_text(str(prompt[key]), max_chars=1200)
        return context

    def _request_node_tools(
        self, stage: str, prompt: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not self.acfg.enable_node_tools or self.acfg.max_node_tool_calls <= 0:
            return []

        tool_prompt = {
            "Introduction": (
                "Before writing the final candidate code, decide whether a few optional "
                "tool observations would reduce uncertainty. Use tools only for concrete "
                "local data-flow, evaluator contract, or API/documentation uncertainty. "
                "Do not request generic EDA; the data and evaluator essentials are already "
                "provided. Return an empty list when no tool is needed."
            ),
            "Available tools": [
                "inspect_api: run a short read-only Python snippet to inspect installed package versions, imports, function signatures, tiny API examples, local split CSV rows, evaluator specs, or image/mask metadata under ./input. No training, writes, package installs, network access, or shell commands.",
                "search_docs: query the local MONAI documentation RAG. Use this only for exact MONAI API, signature, or traceback evidence, not for model choice, augmentation strategy, fine-tuning practice, or pipeline recommendations.",
                "web_search: search the web through the configured Serper/Jina path. Use this only when local context and docs are insufficient for a concrete public library/API question, not for broad strategy or model-selection advice. Never include local sample values, local paths, validation metrics, private split names, or private task text in a web query.",
            ],
            "Tool-use style": (
                "Request only the tools you actually need now. Stable split, sample, "
                "environment, and evaluator facts are already provided in the prompt; "
                "do not re-check them unless the current failure contradicts those facts. "
                f"At most {self.acfg.max_node_tool_calls} tool calls will be executed for this node, "
                f"and at most {NODE_TOOL_CONCURRENCY_LIMIT} node tool calls run globally at once."
            ),
            "Current context": self._tool_decision_context(stage, prompt),
        }
        try:
            response = query(
                system_message=tool_prompt,
                user_message=None,
                func_spec=tool_request_func_spec,
                model=self.acfg.code.model,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Node tool request failed: %s", exc)
            return []
        if not isinstance(response, dict):
            return []
        requests = response.get("requests", [])
        if not isinstance(requests, list):
            return []
        return [request for request in requests if isinstance(request, dict)]

    def _execute_node_tool(
        self, request: dict[str, Any], prompt: dict[str, Any] | None = None
    ) -> str:
        tool = str(request.get("tool", "")).strip()
        if tool == "inspect_api":
            code = str(request.get("code") or "").strip()
            if not code:
                return (
                    "ToolError: inspect_api requires a short Python snippet in `code`."
                )
            from ..tools import python_probe

            return python_probe(code, cwd=self.cfg.workspace_dir)

        query_text = str(request.get("query") or "").strip()
        if not query_text:
            return f"ToolError: {tool} requires a concrete `query`."
        if tool == "search_docs":
            if self._is_local_contract_query(query_text):
                return (
                    "ToolSkipped: evaluator and prediction-contract questions are "
                    "local workspace facts; use the provided evaluator contract or "
                    "inspect_api, not MONAI documentation RAG."
                )
            if not self._is_monai_evidence_query(query_text):
                return (
                    "ToolSkipped: MONAI documentation RAG is reserved for concrete "
                    "MONAI API, signature, or traceback evidence. Use exact names such as "
                    "`monai.transforms.RandRotate` when a MONAI documentation lookup is needed."
                )
            query_text = self._augment_doc_query(query_text, prompt)
            result = self.query_knowledge_base(query_text, topk=3)
            return result or "No local documentation result was returned."
        if tool == "web_search":
            if self._is_local_contract_query(query_text):
                return (
                    "ToolSkipped: evaluator and prediction-contract questions are "
                    "local workspace facts; use the provided evaluator contract or "
                    "inspect_api, not web search."
                )
            result = self.query_web(query_text)
            return result or "No web result was returned or web search is disabled."
        return f"ToolError: unknown tool `{tool}`."

    def _write_tool_trace(self, record: dict[str, Any]) -> None:
        trace_path = self.cfg.log_dir / "tool_traces.jsonl"
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.debug("Failed to write tool trace: %s", exc)
        append_event(
            self.cfg.log_dir,
            "tool_observation",
            stage=record.get("stage"),
            tool=record.get("tool"),
            call_index=record.get("call_index"),
            reason=record.get("reason"),
            query=record.get("query"),
            code_chars=len(str(record.get("code") or "")),
            result_chars=len(str(record.get("result") or "")),
            signal_score=self._tool_signal_score(record),
        )

    def _tool_signal_score(self, record: dict[str, Any]) -> float:
        result = str(record.get("result", ""))
        lowered = result.lower()
        score = 0.0
        if record.get("tool") == "inspect_api":
            score += 3.0
        if "exit_code=0" in lowered:
            score += 1.0
        if "signature" in lowered or "inspect.signature" in str(record.get("code", "")):
            score += 3.0
        if "__version__" in str(record.get("code", "")) or "version" in lowered:
            score += 1.0
        if "traceback" in lowered or "error" in lowered or "exception" in lowered:
            score += 2.0
        if "url:" in lowered or "http://" in lowered or "https://" in lowered:
            score += 1.5
        if "no local documentation result" in lowered or "no web result" in lowered:
            score -= 1.0
        return score

    def _format_tool_record(self, record: dict[str, Any]) -> str:
        code = str(record.get("code") or "")
        query_text = str(record.get("query") or "")
        parts = [
            f"## Tool request {record.get('call_index')}",
            f"tool: {record.get('tool')}",
            f"reason: {truncate_text(str(record.get('reason', '')), max_chars=300)}",
        ]
        if query_text:
            parts.append(f"query: {truncate_text(query_text, max_chars=300)}")
        if code:
            parts.extend(
                [
                    "code:",
                    wrap_code(
                        truncate_text(
                            code,
                            max_chars=900,
                            marker="probe code trimmed",
                        ),
                        lang="python",
                    ),
                ]
            )
        parts.extend(
            [
                "observation:",
                truncate_text(
                    str(record.get("result", "")),
                    max_chars=TOOL_OBSERVATION_ENTRY_CHARS,
                    marker="tool observation trimmed",
                ),
            ]
        )
        return "\n".join(parts)

    def _compact_tool_observations(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return ""

        formatted = [(record, self._format_tool_record(record)) for record in records]
        all_text = "\n\n".join(text for _, text in formatted)
        if len(all_text) <= TOOL_OBSERVATION_TOTAL_CHARS:
            return all_text

        ranked_records = sorted(
            records,
            key=lambda record: (
                self._tool_signal_score(record),
                int(record.get("call_index", 0)),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        used = 0
        for record in ranked_records:
            text = self._format_tool_record(record)
            if used + len(text) + 2 > TOOL_OBSERVATION_TOTAL_CHARS:
                continue
            selected.append(record)
            used += len(text) + 2
        selected.sort(key=lambda record: int(record.get("call_index", 0)))
        omitted = len(records) - len(selected)
        header = (
            f"Tool observations are compacted by signal; full raw results are saved in "
            f"`{self.cfg.log_dir / 'tool_traces.jsonl'}`."
        )
        if omitted > 0:
            header += f" Omitted {omitted} lower-signal observation(s) from prompt."
        return (
            header
            + "\n\n"
            + "\n\n".join(self._format_tool_record(record) for record in selected)
        )

    def generate_draft_plan_batch(self, count: int) -> list[str]:
        """Generate textual bootstrap plans for root drafts."""
        count = max(0, int(count))
        if count == 0:
            return []
        if self.data_summary is None:
            self.update_data_summary()
        prompt = {
            "Introduction": (
                f"Analyze the task description, data/evaluator facts, and any "
                f"existing root-draft plan summaries. Create {count} concise textual "
                "root-draft plans for BioMaker. These plans will later be assigned "
                "to separate parallel code-generation workers."
            ),
            "Task Description": self.context_builder._task_description_value(),
            "Data and Evaluation Summary": self.get_data_summary(),
            "Existing Root Draft Plan Summaries": self._root_draft_plan_summaries(),
            "Planning requirements": [
                "Return exactly the requested number of plans when possible.",
                "Each plan must be a text-only strategy description, not code or pseudocode.",
                "Base each plan on the provided task, split, input, and evaluator facts.",
                "Let the task/data/evaluator facts determine the strategy; do not satisfy arbitrary category quotas.",
                "Make plans substantively different in their task-relevant idea or evidence path, not by forcing arbitrary categories.",
                "Avoid duplicating existing root draft plans.",
                "Use the provided data/evaluator facts instead of asking for more exploration.",
            ],
        }
        try:
            response = query(
                system_message=prompt,
                user_message=None,
                func_spec=draft_plan_batch_func_spec,
                model=self.acfg.code.model,
                temperature=max(float(self.acfg.code.temp or 0.0), 0.7),
            )
        except Exception as exc:
            logger.warning("Draft plan batch generation failed: %s", exc)
            response = {}
        plan_items = response.get("plans", []) if isinstance(response, dict) else []
        plans: list[str] = []
        for item in plan_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            plan = str(item.get("plan") or "").strip()
            text = f"{name}: {plan}" if name else plan
            if text and text not in plans:
                plans.append(text)
            if len(plans) >= count:
                break
        while len(plans) < count:
            plans.append(
                f"Generated plan {len(plans) + 1}: use the task description, "
                "agent-visible training split, and frozen evaluator facts to define "
                "one complete root-draft strategy that can later be implemented as a "
                "single runnable candidate."
            )
        append_event(
            self.cfg.log_dir,
            "draft_plan_batch_generated",
            requested=count,
            returned=len(plans),
            plan_previews=[truncate_text(plan, max_chars=240) for plan in plans],
        )
        return plans

    def _root_draft_plan_summaries(self) -> str:
        drafts = sorted(
            self.journal.draft_nodes,
            key=lambda node: (
                node.step is None,
                node.step if node.step is not None else 10**9,
                node.ctime,
            ),
        )
        if not drafts:
            return "No existing root draft plans."

        lines = [
            "Existing root draft plans and outcomes. Code is intentionally omitted."
        ]
        for node in drafts:
            if node.is_buggy is True:
                result = "bug"
            elif node.is_buggy is False:
                result = "success"
            else:
                result = "not_scored"
            metric = (
                "n/a"
                if node.metric is None or node.metric.value is None
                else f"{float(node.metric.value):.4f}"
            )
            step = "not_scored" if node.step is None else str(node.step)
            lines.append(
                "\n".join(
                    [
                        f"- step: {step}",
                        f"  id: {node.id[:8]}",
                        f"  result: {result}",
                        f"  validation_metric: {metric}",
                        f"  plan: {node.plan or ''}",
                    ]
                )
            )
        return "\n".join(lines)

    def select_round_tasks(self) -> list[RoundTask]:
        tasks = super().select_round_tasks()
        if not tasks or any(task.parent is not None for task in tasks):
            return tasks
        plans = self.generate_draft_plan_batch(len(tasks))
        return [
            RoundTask(
                parent=task.parent,
                selection_reason=(
                    f"bootstrap root draft plan {idx}/{len(tasks)}: {plans[idx - 1]}"
                ),
                draft_plan=plans[idx - 1],
            )
            for idx, task in enumerate(tasks, start=1)
        ]

    def attach_node_tool_observations(self, stage: str, prompt: dict[str, Any]) -> None:
        records: list[dict[str, Any]] = []
        seen_requests: set[tuple[str, str, str]] = set()
        call_count = 0
        max_calls = max(1, int(self.acfg.max_node_tool_calls))
        requests = self._request_node_tools(stage, prompt)
        for request in requests:
            tool_name = str(request.get("tool", ""))
            request_key = (
                tool_name,
                (
                    ""
                    if tool_name == "inspect_api"
                    else str(request.get("query", "")).strip()
                ),
                str(request.get("code", "")).strip(),
            )
            if request_key in seen_requests:
                continue
            seen_requests.add(request_key)
            if call_count >= max_calls:
                records.append(
                    {
                        "stage": stage,
                        "call_index": call_count + 1,
                        "tool": "budget",
                        "reason": "Configured safety budget reached.",
                        "query": "",
                        "code": "",
                        "result": "NodeToolBudget: stopped tool use after the configured safety budget.",
                    }
                )
                break
            call_count += 1
            with _NODE_TOOL_SEMAPHORE:
                result = self._execute_node_tool(request, prompt=prompt)
            record = {
                "stage": stage,
                "call_index": call_count,
                "tool": request.get("tool"),
                "reason": request.get("reason", ""),
                "query": (
                    self._augment_doc_query(str(request.get("query", "")), prompt)
                    if request.get("tool") == "search_docs"
                    else request.get("query", "")
                ),
                "code": request.get("code", ""),
                "result": result,
            }
            records.append(record)
            self._write_tool_trace(record)

        if records:
            logger.debug("Attached %d node tool observation(s)", call_count)
            prompt["Tool Observations"] = self._compact_tool_observations(records)

    def _draft(self) -> Node:
        prompt = self.context_builder.build_draft()
        self.attach_node_tool_observations("draft", prompt)
        prompt = self.context_builder._ordered_prompt(prompt)
        append_event(
            self.cfg.log_dir,
            "context_built",
            stage="draft",
            shape=prompt_shape(prompt),
        )
        plan, code, prompt_capture = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            prompt=prompt_capture["prompt"],
            prompt_messages=prompt_capture["messages"],
            prompt_stats=prompt_capture["stats"],
        )

    def _debug(self, parent_node: Node) -> Node:
        prompt = self.context_builder.build_debug(parent_node)
        self.attach_node_tool_observations("debug", prompt)
        prompt = self.context_builder._ordered_prompt(prompt)
        append_event(
            self.cfg.log_dir,
            "context_built",
            stage="debug",
            parent_id=parent_node.id,
            shape=prompt_shape(prompt),
        )
        plan, code, prompt_capture = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
            prompt=prompt_capture["prompt"],
            prompt_messages=prompt_capture["messages"],
            prompt_stats=prompt_capture["stats"],
        )

    def _improve(self, parent_node: Node) -> Node:
        prompt = self.context_builder.build_improve(parent_node)
        self.attach_node_tool_observations("improve", prompt)
        prompt = self.context_builder._ordered_prompt(prompt)
        append_event(
            self.cfg.log_dir,
            "context_built",
            stage="improve",
            parent_id=parent_node.id,
            shape=prompt_shape(prompt),
        )
        plan, code, prompt_capture = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
            prompt=prompt_capture["prompt"],
            prompt_messages=prompt_capture["messages"],
            prompt_stats=prompt_capture["stats"],
        )

    def generate_node(self, task: RoundTask) -> Node:
        """Generate plan/code for one selected tree task without executing it."""
        if not self.journal.nodes or self.data_summary is None:
            self.update_data_summary()

        parent_node = task.parent
        selection_reason = task.selection_reason
        self._last_selection_reason = selection_reason
        logger.debug(
            "Agent is generating code, parent node type: %s", type(parent_node)
        )
        append_event(
            self.cfg.log_dir,
            "node_selected",
            stage=(
                "draft"
                if parent_node is None
                else "debug" if parent_node.is_buggy else "improve"
            ),
            parent_id=None if parent_node is None else parent_node.id,
            selection_reason=selection_reason,
        )

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)
        result_node.selection_reason = selection_reason
        return result_node

    def step(self, exec_callback: ExecCallbackType):
        tasks = self.select_round_tasks()
        if not tasks:
            raise RuntimeError("No proposal task is available.")
        task = tasks[0]
        result_node = self.generate_node(task)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True, node=result_node),
        )
        self.journal.append(result_node)
        return result_node
