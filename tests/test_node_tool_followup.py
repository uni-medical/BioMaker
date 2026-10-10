import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import biomaker.agents.base_agent as base_agent
import biomaker.agents.proposal_agent as proposal_agent
from biomaker.agents.proposal_agent import ProposalAgent, TOOL_OBSERVATION_TOTAL_CHARS
from biomaker.journal import Journal
from biomaker.monai_rag.query_rag_db import MED_KNOWLEDGE_RAG
from biomaker.utils.response import truncate_text


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        base_agent, "environment_manifest_text", lambda **kwargs: "test environment"
    )
    cfg = SimpleNamespace(
        workspace_dir=tmp_path,
        log_dir=tmp_path,
        exec=SimpleNamespace(include_medical_packages=True),
        agent=SimpleNamespace(
            data_summary=False,
            enable_node_tools=True,
            max_node_tool_calls=4,
            code=SimpleNamespace(model="test-model", temp=0.7),
        ),
    )
    instance = ProposalAgent("test task", cfg, Journal())
    instance.data_summary = ""
    return instance


def request(code, reason="inspect API"):
    return {"tool": "inspect_api", "reason": reason, "code": code}


def response(*requests):
    return {"requests": list(requests)}


def trace_records(agent):
    path = agent.cfg.log_dir / "tool_traces.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_followup_selection_receives_actual_observation(agent, monkeypatch):
    marker = "FIRST_TOOL_RESULT_IN_THE_MIDDLE"
    result = "a" * 600 + marker + "b" * 600
    first_request = request("#" + "c" * 899, reason="r" * 300)
    query = Mock(side_effect=[response(first_request), response()])
    execute = Mock(return_value=result)
    monkeypatch.setattr(proposal_agent, "query", query)
    monkeypatch.setattr(agent, "_execute_node_tool", execute)
    prompt = {"Memory": "m" * 1800}

    agent.attach_node_tool_observations("improve", prompt)

    assert query.call_count == 2
    context = query.call_args_list[1].kwargs["system_message"]["Current context"]
    observations = context["Tool Observations"]
    assert marker in observations
    assert marker not in truncate_text(observations, max_chars=1200)
    assert "remaining budget: 3" in observations
    assert len(context["Memory"]) <= 1200
    assert marker in prompt["Tool Observations"]
    assert trace_records(agent)[0]["result"] == result


def test_tool_observation_context_keeps_existing_total_cap(agent):
    context = agent._tool_decision_context(
        "debug",
        {
            "Tool Observations": "o" * (TOOL_OBSERVATION_TOTAL_CHARS + 1000),
            "Memory": "m" * 2000,
        },
    )

    assert len(context["Tool Observations"]) <= TOOL_OBSERVATION_TOTAL_CHARS
    assert len(context["Memory"]) <= 1200


def test_rag_passages_reach_initial_plan_prompt(agent, tmp_path, monkeypatch):
    index_dir = tmp_path / "rag"
    index_dir.mkdir()
    documents = [
        {
            "page_content": (
                "Cache training images. "
                + f"Unique documentation passage {index}. " * 60
            ),
            "metadata": {
                "source": f"guide-{index}.html",
                "title": f"Caching guide {index}",
                "version": "test",
            },
        }
        for index in range(5)
    ]
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(document) for document in documents), encoding="utf-8"
    )
    agent.acfg.enable_monai_knowledge_base = True
    agent.acfg.enable_query2doc = False
    agent.acfg.knowledge_base = SimpleNamespace(
        model="test-model", retrieval_mode="keyword", topk=5, required=False
    )
    agent.knowledge_base = MED_KNOWLEDGE_RAG(persist_dir=index_dir)
    monkeypatch.setattr(
        agent.context_builder,
        "build_draft",
        lambda: {"Task Description": "test task", "Instructions": {}},
    )
    docs_request = {
        "tool": "search_docs",
        "query": "how to cache training images",
        "reason": "Find documentation before choosing a data pipeline.",
    }
    query = Mock(
        side_effect=[
            response(docs_request),
            response(),
            {"plans": [{"name": "Cached input", "plan": "Use documented caching."}]},
        ]
    )
    monkeypatch.setattr(proposal_agent, "query", query)

    plans = agent.generate_draft_plan_batch(1)

    assert plans == ["Cached input: Use documented caching."]
    assert (
        query.call_args_list[0].kwargs["system_message"]["Current context"]["Stage"]
        == "plan"
    )
    assert (
        query.call_args.kwargs["func_spec"] is proposal_agent.draft_plan_batch_func_spec
    )
    observations = query.call_args.kwargs["system_message"]["Tool Observations"]
    assert len(observations) > 1600
    for index in range(5):
        assert f"Source: guide-{index}.html" in observations
        assert f"Title: Caching guide {index}" in observations
        assert f"Unique documentation passage {index}" in observations
    followup = query.call_args_list[1].kwargs["system_message"]["Current context"]
    assert followup["Tool Observations"] == observations
    traces = trace_records(agent)
    assert len(traces) == 1
    assert traces[0]["stage"] == "plan"
    assert traces[0]["query"] == docs_request["query"]


def test_search_docs_uses_the_requested_query_without_unrelated_context(
    agent, monkeypatch
):
    retrieve = Mock(return_value="Source: docs\nSnippet: useful documentation")
    monkeypatch.setattr(agent, "query_knowledge_base", retrieve)

    result = agent._execute_node_tool(
        {"tool": "search_docs", "query": "how to cache training images"},
        prompt={"Previous solution": "monai.losses.DiceLoss monai.networks.nets.UNet"},
    )

    retrieve.assert_called_once_with("how to cache training images")
    assert "useful documentation" in result


def test_large_rag_result_keeps_complete_passages_within_context_budget(agent):
    passages = [
        f"Source: guide-{index}.html\nTitle: Guide {index}\nSnippet: "
        + "documentation " * 100
        + f"\nEnd of passage {index}."
        for index in range(20)
    ]
    record = {
        "tool": "search_docs",
        "query": "caching examples",
        "result": "\n\n---\n\n".join(passages),
    }

    observations = agent._compact_tool_observations([record])

    assert "Source: guide-0.html" in observations
    assert "Additional documentation omitted" in observations
    for index in range(20):
        if f"Source: guide-{index}.html" in observations:
            assert f"End of passage {index}." in observations
    prompt = {"Tool Observations": "Node tool calls used: 1/4.\n\n" + observations}
    context = agent._tool_decision_context("debug", prompt)
    assert context["Tool Observations"] == prompt["Tool Observations"]
    assert len(context["Tool Observations"]) <= TOOL_OBSERVATION_TOTAL_CHARS


def test_followup_shares_budget_deduplication_and_trace(agent, monkeypatch):
    query = Mock(
        side_effect=[
            response(request("a"), request(" a ", "duplicate"), request("b")),
            response(
                request("a", "still duplicate"),
                request("c"),
                request("d"),
                request("e"),
            ),
        ]
    )
    execute = Mock(side_effect=lambda req, **kwargs: "result " + req["code"])
    monkeypatch.setattr(proposal_agent, "query", query)
    monkeypatch.setattr(agent, "_execute_node_tool", execute)
    prompt = {}

    agent.attach_node_tool_observations("debug", prompt)

    assert query.call_count == 2
    assert execute.call_count == 4
    assert [call.args[0]["code"] for call in execute.call_args_list] == [
        "a",
        "b",
        "c",
        "d",
    ]
    records = trace_records(agent)
    assert [record["call_index"] for record in records] == [1, 2, 3, 4]
    assert [record["result"] for record in records] == [
        "result a",
        "result b",
        "result c",
        "result d",
    ]
    context = query.call_args_list[1].kwargs["system_message"]["Current context"]
    assert "result a" in context["Tool Observations"]
    assert "result b" in context["Tool Observations"]
    assert "remaining budget: 2" in context["Tool Observations"]
    assert "NodeToolBudget" in prompt["Tool Observations"]


@pytest.mark.parametrize(
    "enabled,budget,first_requests,selection_count,execution_count",
    [
        (False, 4, [], 0, 0),
        (True, 0, [], 0, 0),
        (True, -1, [], 0, 0),
        (True, 4, [], 1, 0),
        (True, 4, [request(str(index)) for index in range(4)], 1, 4),
    ],
)
def test_followup_skips_unnecessary_selection(
    agent,
    monkeypatch,
    enabled,
    budget,
    first_requests,
    selection_count,
    execution_count,
):
    agent.acfg.enable_node_tools = enabled
    agent.acfg.max_node_tool_calls = budget
    query = Mock(return_value=response(*first_requests))
    execute = Mock(return_value="observed")
    monkeypatch.setattr(proposal_agent, "query", query)
    monkeypatch.setattr(agent, "_execute_node_tool", execute)
    prompt = {}

    agent.attach_node_tool_observations("draft", prompt)

    assert query.call_count == selection_count
    assert execute.call_count == execution_count
    if execution_count:
        assert len(trace_records(agent)) == execution_count
    else:
        assert "Tool Observations" not in prompt
        assert not (agent.cfg.log_dir / "tool_traces.jsonl").exists()


def test_followup_selector_failure_preserves_first_result(agent, monkeypatch):
    query = Mock(side_effect=[response(request("a")), RuntimeError("selector failed")])
    execute = Mock(return_value="first observation")
    monkeypatch.setattr(proposal_agent, "query", query)
    monkeypatch.setattr(agent, "_execute_node_tool", execute)
    prompt = {}

    agent.attach_node_tool_observations("improve", prompt)

    assert query.call_count == 2
    assert execute.call_count == 1
    assert "first observation" in prompt["Tool Observations"]
    assert trace_records(agent)[0]["result"] == "first observation"


def test_tool_error_is_observed_without_automatic_retry(agent, monkeypatch):
    query = Mock(side_effect=[response(request("a")), response(request("a", "retry"))])
    execute = Mock(return_value="ToolError: missing API")
    monkeypatch.setattr(proposal_agent, "query", query)
    monkeypatch.setattr(agent, "_execute_node_tool", execute)
    prompt = {}

    agent.attach_node_tool_observations("debug", prompt)

    assert query.call_count == 2
    assert execute.call_count == 1
    context = query.call_args_list[1].kwargs["system_message"]["Current context"]
    assert "ToolError: missing API" in context["Tool Observations"]
    assert len(trace_records(agent)) == 1


def test_tool_selection_stops_after_two_rounds_with_budget_left(agent, monkeypatch):
    query = Mock(side_effect=[response(request("a")), response(request("b"))])
    execute = Mock(return_value="observed")
    monkeypatch.setattr(proposal_agent, "query", query)
    monkeypatch.setattr(agent, "_execute_node_tool", execute)
    prompt = {}

    agent.attach_node_tool_observations("improve", prompt)

    assert query.call_count == 2
    assert execute.call_count == 2
    assert "remaining budget: 2" in prompt["Tool Observations"]


def test_tool_execution_exceptions_keep_existing_behavior(agent, monkeypatch):
    query = Mock(return_value=response(request("a")))
    execute = Mock(side_effect=RuntimeError("tool execution failed"))
    monkeypatch.setattr(proposal_agent, "query", query)
    monkeypatch.setattr(agent, "_execute_node_tool", execute)

    with pytest.raises(RuntimeError, match="tool execution failed"):
        agent.attach_node_tool_observations("debug", {})

    assert query.call_count == 1
    assert execute.call_count == 1
