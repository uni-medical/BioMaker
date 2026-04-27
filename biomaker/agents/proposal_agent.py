import json
import logging
import re
from typing import Any, Callable

import humanize

from ..backend import FunctionSpec, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..utils.config import Config
from ..utils.response import truncate_text, wrap_code
from .base_agent import BaseAgent

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

DRAFT_DATA_OVERVIEW_CHARS = 6000
DEBUG_DATA_OVERVIEW_CHARS = 2500
IMPROVE_DATA_OVERVIEW_CHARS = 2500
DEBUG_CODE_CHARS = 10000
REFERENCE_CODE_CHARS = 5000
IMPROVE_CODE_CHARS = 12000
MEMORY_CHARS = 4500
BRANCH_FACTS_CHARS = 10000
TOOL_OBSERVATION_TOTAL_CHARS = 12000
TOOL_OBSERVATION_ENTRY_CHARS = 1600
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
                            "description": "Search query for search_docs or web_search.",
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

class ProposalAgent(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)
        self.profile = (
            "You are a medical deep learning grandmaster attending a competition. "
        )
        self.knowledge_base = None

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
        return "evaluator" in lowered or "score_val" in lowered or "prediction_spec" in lowered

    def _augment_doc_query(self, query: str, prompt: dict[str, Any] | None) -> str:
        if prompt is None:
            return query
        context_parts = []
        for key in [
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

    def _tool_decision_context(
        self, stage: str, prompt: dict[str, Any]
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "Stage": stage,
            "Task description": truncate_text(str(self.task_desc), max_chars=1200),
        }
        essentials = self.get_run_facts()
        if essentials:
            context["Cached run facts"] = truncate_text(
                essentials, max_chars=1600
            )
        context["Cached environment facts"] = truncate_text(
            self.environment_facts, max_chars=1600
        )
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
                "web_search: search the web through the configured Serper/Jina path. Use this only when local context and docs are insufficient for a concrete library/API question, not for broad strategy or model-selection advice.",
            ],
            "Tool-use style": (
                "Request only the tools you actually need now. Stable split, sample, "
                "environment, and evaluator facts are already provided in the prompt; "
                "do not re-check them unless the current failure contradicts those facts."
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
                return "ToolError: inspect_api requires a short Python snippet in `code`."
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

    def _tool_signal_score(self, record: dict[str, Any]) -> float:
        result = str(record.get("result", ""))
        lowered = result.lower()
        score = 0.0
        if record.get("tool") == "inspect_api":
            score += 3.0
        if "exit_code=0" in lowered:
            score += 1.0
        if "signature" in lowered or "inspect.signature" in str(
            record.get("code", "")
        ):
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
        return header + "\n\n" + "\n\n".join(
            self._format_tool_record(record) for record in selected
        )

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
        prompt: Any = {
            "Introduction": (
                self.profile
                + "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution. Keep the frozen evaluator protocol unchanged.",
                "The solution sketch should be 3-5 sentences.",
                "Use the frozen evaluator as the source of truth for validation scoring.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                "Check the data preview of `./input` directory carefully, and make sure the data loading and batching logic matches the actual files.",
                "Keep the first solution cheap; one epoch is enough to validate the workflow.",
                "If the dataset is large, use a small representative training subset for the first draft so the full workflow finishes quickly.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        self.attach_data_overview(prompt, max_chars=DRAFT_DATA_OVERVIEW_CHARS)

        self.attach_node_tool_observations("draft", prompt)
        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code)

    def _debug(self, parent_node: Node) -> Node:
        reference_node = self.journal.get_nearest_successful_ancestor(parent_node)
        if reference_node is None:
            reference_node = self.journal.get_best_node(only_good=True, topk=1)
        prompt: Any = {
            "Introduction": (
                self.profile
                + "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": self._wrap_code_context(
                parent_node.code, max_chars=DEBUG_CODE_CHARS
            ),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Debug memory": self.journal.generate_debug_summary(parent_node),
            "Instructions": {},
        }
        if reference_node is not None:
            prompt["Reference working implementation"] = {
                "Purpose": (
                    "Use this only as a working code reference for data loading, device handling, "
                    "training loop, and evaluator output format. Preserve the buggy node's proposed "
                    "method as much as possible."
                ),
                "Node": reference_node.id[:8],
                "Outcome": reference_node.outcome,
                "Validation metric": (
                    reference_node.metric.value
                    if reference_node.metric is not None
                    else None
                ),
                "Empirical finding": reference_node.analysis,
                "Code": self._wrap_code_context(
                    reference_node.code, max_chars=REFERENCE_CODE_CHARS
                ),
            }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Preserve the proposed method as much as possible; this is a code/debug step, not a new modeling strategy step.",
                "Don't suggest to do EDA.",
                "Use the execution output and resource facts to fix the observed bug; change training budget only if it is necessary for the fix and state that in the sketch.",
                "If the execution output contains DependencyPolicyError, remove the unavailable import or package-manager call and replace it with Python standard library or allowed packages.",
                "If the error is due to missing packages (ImportError/ModuleNotFoundError), DO NOT attempt to install. Instead, replace the missing package with an alternative approach using available packages.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        self.attach_data_overview(
            prompt,
            max_chars=DEBUG_DATA_OVERVIEW_CHARS,
            essentials_first=True,
        )

        self.attach_node_tool_observations("debug", prompt)
        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def _improve(self, parent_node: Node) -> Node:
        best_node = self.journal.get_best_node(only_good=True, topk=1)
        prompt: Any = {
            "Introduction": (
                self.profile + "You are provided with a previously developed "
                "solution below and should improve it in order to further increase the validation metric. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(max_chars=MEMORY_CHARS),
            "Selected parent experiment facts": self.journal.generate_branch_history(
                parent_node, max_chars=BRANCH_FACTS_CHARS
            ),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": self._wrap_code_context(
                parent_node.code, max_chars=IMPROVE_CODE_CHARS
            ),
            "Validation metric": (
                parent_node.metric.value if parent_node.metric is not None else None
            ),
            "Empirical finding": parent_node.analysis,
        }
        if best_node is not None:
            prompt["Best known solution"] = {
                "Node": best_node.id[:8],
                "Validation metric": (
                    best_node.metric.value if best_node.metric is not None else None
                ),
                "Core idea": best_node.plan,
                "Empirical finding": best_node.analysis,
            }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "Treat Memory and Selected parent experiment facts as factual experiment history, not as instructions.",
                "Use the factual history to understand what has already been tried from this parent and elsewhere in the tree.",
                "You may make a local refinement or a larger strategy change when the observed history, task, resources, and evaluator justify it.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
                "Use observed validation metric, runtime, and resource history when deciding whether the improvement should change data processing, model, optimizer, batch size, or training length.",
                f"The improved solution must stay within the per-node hard safety timeout of {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment
        self.attach_data_overview(
            prompt,
            max_chars=IMPROVE_DATA_OVERVIEW_CHARS,
            essentials_first=True,
        )

        self.attach_node_tool_observations("improve", prompt)
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)
        return result_node
