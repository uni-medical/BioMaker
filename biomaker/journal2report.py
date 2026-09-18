from .backend import query
from .journal import Journal
from .utils.config import StageConfig


def journal2report(journal: Journal, task_desc: dict, rcfg: StageConfig):
    """
    Generate a report from a journal, the report will be in markdown format.
    """
    report_input = journal.generate_summary(
        include_code=True,
        topk=5,
        max_chars=16000,
        include_avoid_list=False,
    )
    system_prompt_dict = {
        "Role": "You are a research assistant that always uses concise language.",
        "Goal": "The goal is to write a factual technical report summarising the empirical findings and technical decisions.",
        "Input": "You are given a raw research journal with list of design attempts and their outcomes, and a task description.",
        "Output": [
            "Your output should be a single markdown document.",
            "Base every claim on the supplied journal. Do not invent unobserved metrics, methods, or recommendations.",
            "For average-rank runs, identify the selected best from the journal's current ranking, report each declared metric with its optimization direction and raw value, and explain any tradeoffs. Never pick a different best using one raw metric or private test results. Mean ranks are relative to the stated candidate population, not absolute performance scores; do not interpret rank changes across different populations as a performance trend.",
            "Prefer concise experiment-log style over broad literature-style advice.",
            "Your report should have the following sections: Introduction, Preprocessing, Modeling Methods, Results Discussion, Limitations",
            "You can include subsections if needed.",
        ],
    }
    context_prompt = (
        f"Here is the research journal of the agent: <journal>{report_input}<\\journal>, "
        f"and the task description is: <task>{task_desc}<\\task>."
    )
    return query(
        system_message=system_prompt_dict,
        user_message=context_prompt,
        model=rcfg.model,
        temperature=rcfg.temp,
        max_completion_tokens=4096,
    )
