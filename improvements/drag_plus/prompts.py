"""Every baseline prompt string, verbatim, as small pure builders.

Text must not drift: the golden transcript (T0) compares the rendered prompts with the
frozen ``model/drag.py`` character for character. The StrategyQA branches are kept as the
baseline has them, including the missing space before ``Question:`` in the debate message.
"""
from __future__ import annotations

from typing import Optional

PROCEED = "Proceed with your role instructions."
_EVIDENCE_ABOVE = "The evidence is in the shared context above. "

PROPONENT_QUERY = (
    "You are a debater. Argue that the current retrieved content is sufficient to answer "
    "the question and no further retrieval is needed. Deliver a brief, strong argument "
    "with clear reasoning. Do not suggest further retrieval. No extra explanations."
)

OPPONENT_QUERY = """You are a critical thinker and debater, and your task is to challenge the sufficiency of the current retrieved content. Argue that the current information is insufficient to generate a reliable answer and propose either query optimization or query expansion.
The action you can choose:
1. Query Optimization: If the retrieved content is somewhat relevant but has expression or scope issues, improve the query using this format: Query Optimization: [Original Query] -> [New Query].
2. Query Expansion: If critical information is missing, propose a new query using this format: Query Expansion: [New Query].
Deliver a brief, strong argument with clear reasoning, then you must choose only one action. The output must be in the exact format after your reasoning, without additional explanation, and keep the new query short and precise."""

JUDGE_RETRY = (
    "Your previous output was not a valid single-agent verdict. "
    "Evaluate the same arguments and select exactly one winner. "
    "Reply with exactly Proponent or Opponent. "
    "Do not list both names, add punctuation, or explain."
)

_ANSWER_FORMAT = (
    "Explain your answer, and always put the answer after 'The answer is: ', "
    "e.g.'The answer is: answer.', at the end of your response."
)
_ANSWER_FORMAT_YES_NO = (
    "Given two answer candidates, Yes and No, choose the best answer choice. "
    "Explain your answer, and always put the answer after 'The answer is: ', "
    "e.g.'The answer is: Yes.', at the end of your response."
)
_CONCISE_ANSWER = (
    "Use at most 2 sentences, then put the answer after 'The answer is: ', "
    "e.g.'The answer is: answer.'."
)
_DEBATE_INTRO = (
    "I will give the answers and arguments to this question from other agents. "
    "Use their solution as additional advice; note that they may be wrong. "
)
_MODERATOR_INTRO = (
    "You are a moderator in a debate competition. Your task is to determine the correct "
    "final answer based on the arguments presented by debaters. "
)
_FINAL_ONLY = "Output only the final answer with no explanations or additional text."
_YES_NO_CHOICE = "Given two answer candidates, Yes and No, choose the best answer choice. "
_JUDGE_INTRO_HEAD = (
    "You are the judge in a debate. Your task is to evaluate the arguments from agents. "
    "There are two types of agents:\n"
)
_JUDGE_AGENT_LINES = {
    "Proponent": "Argues that the current retrieved content is sufficient.",
    "Opponent": ("Argues that the current retrieved content is insufficient and "
                 "proposes query refinement."),
}
_JUDGE_INTRO = (
    _JUDGE_INTRO_HEAD
    + "1. Proponent Agent: " + _JUDGE_AGENT_LINES["Proponent"] + "\n"
    + "2. Opponent Agent: " + _JUDGE_AGENT_LINES["Opponent"] + "\n"
)
_JUDGE_TAIL = "Output only the agent's name. No explanations."


def _with_prefix(content: str, prefix: Optional[str]) -> str:
    """Append the unchanged role instruction after a shared evidence-first prefix."""
    return content if prefix is None else f"{prefix}{content}"


def _answer_format(strategy_qa: bool, concise: bool) -> str:
    """Agent answer-format text; ``concise`` keeps the same answer marker."""
    if not concise:
        return _ANSWER_FORMAT_YES_NO if strategy_qa else _ANSWER_FORMAT
    return (_YES_NO_CHOICE + _CONCISE_ANSWER) if strategy_qa else _CONCISE_ANSWER


def format_reference(retrieval_result) -> str:
    """Render one query's passages in the baseline ``Doc i(Title: t) text`` format."""
    out = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        out += f"Doc {idx+1}(Title: {title}) {text}\n"
    return out


def format_query_pool(query_pool) -> str:
    """Render the pool; an evidence-aware pool supplies its own labelled layout."""
    render = getattr(query_pool, "format_evidence", None)
    if callable(render):
        return render()
    out = ""
    for i, query in enumerate(query_pool):
        out += f"Query {i+1}: {query}\nRetrieved Content: {format_reference(query_pool[query])}"
    return out


def query_agent_system(agent_name: str, prefix: Optional[str] = None) -> dict:
    if "Proponent" in agent_name:
        return {"role": "system", "content": _with_prefix(PROPONENT_QUERY, prefix)}
    if "Opponent" in agent_name:
        return {"role": "system", "content": _with_prefix(OPPONENT_QUERY, prefix)}
    raise ValueError("The agent name must contain either Proponent or Opponent")


def query_round_user(question: str, query_pool, prefix: Optional[str] = None) -> dict:
    if prefix is not None:
        return {"role": "user", "content": PROCEED}
    return {"role": "user", "content": f"Question: {question}\n{format_query_pool(query_pool)}"}


def _agent_role(agent: str) -> str:
    """Map a full agent name to the role label used by the Judge intro."""
    if "Proponent" in agent:
        return "Proponent"
    if "Opponent" in agent:
        return "Opponent"
    raise ValueError("The agent name must contain either Proponent or Opponent")


def _judge_intro(agent_order) -> str:
    """The baseline intro, or the same lines renumbered in ``agent_order`` (J12)."""
    if agent_order is None:
        return _JUDGE_INTRO
    roles = (_agent_role(agent) for agent in agent_order)
    lines = [f"{index}. {role} Agent: {_JUDGE_AGENT_LINES[role]}"
             for index, role in enumerate(roles, 1)]
    return _JUDGE_INTRO_HEAD + "\n".join(lines) + "\n"


def query_moderator_system(agent_outputs: dict, input_query: str, query_pool,
                           prefix: Optional[str] = None,
                           agent_order=None) -> dict:
    """``agent_outputs`` maps agent name to its raw output for the current round.

    ``agent_order`` (J12) lists the agents in presentation order; None keeps the
    baseline Proponent-then-Opponent intro and argument order byte for byte.
    """
    order = tuple(agent_outputs) if agent_order is None else tuple(agent_order)
    arguments = "".join(f"{agent}: {agent_outputs[agent]}\n" for agent in order)
    body = _judge_intro(agent_order)
    if prefix is None:
        body += f"Question: {input_query}\n{format_query_pool(query_pool)}\n"
    body += f"Agents Arguments:\n{arguments}\n{_JUDGE_TAIL}"
    return {"role": "system", "content": _with_prefix(body, prefix)}


def judge_retry_user() -> dict:
    return {"role": "user", "content": JUDGE_RETRY}


def question_user(question: str, prefix: Optional[str] = None) -> dict:
    if prefix is not None:
        return {"role": "user", "content": PROCEED}
    return {"role": "user", "content": f"Question: {question}\n"}


def answer_agent_system(agent_name: str, query_pool, strategy_qa: bool,
                        concise: bool = False, prefix: Optional[str] = None) -> dict:
    """Grounded Proponent or closed-book Opponent system message."""
    fmt = _answer_format(strategy_qa, concise)
    if "Proponent" in agent_name:
        docs = (_EVIDENCE_ABOVE if prefix is not None
                else f"The following are given documents.\n{format_query_pool(query_pool)}")
        content = f"Answer the question based on the given document. {fmt} {docs}"
        return {"role": "system", "content": _with_prefix(content, prefix)}
    if "Opponent" in agent_name:
        content = f"Answer the question based on your own knowledge. {fmt}"
        return {"role": "system", "content": _with_prefix(content, prefix)}
    raise ValueError("The agent name must contain either Proponent or Opponent")


def answer_only_system(query_pool, strategy_qa: bool,
                       prefix: Optional[str] = None) -> dict:
    choice = _YES_NO_CHOICE if strategy_qa else ""
    docs = "" if prefix is not None else f"\n{format_query_pool(query_pool)}"
    content = f"Answer the question based on the given document. {choice}{_FINAL_ONLY}{docs}"
    return {"role": "system", "content": _with_prefix(content, prefix)}


def answer_debate_user(others: dict, question: str, round_index: int,
                       strategy_qa: bool, concise: bool = False,
                       prefix: Optional[str] = None) -> dict:
    """``others`` maps agent name to its chat history; round r reads index 2r."""
    fmt = _answer_format(strategy_qa, concise)
    asked = "" if prefix is not None else f"Question: {question}\n"
    text = f"{_DEBATE_INTRO}{fmt}{asked}Other agents responses:\n"
    for agent_name, history in others.items():
        text += f"{agent_name}: {history[round_index*2]['content']}\n"
    return {"role": "user", "content": text}


def answer_moderator_system(strategy_qa: bool, prefix: Optional[str] = None) -> dict:
    choice = _YES_NO_CHOICE if strategy_qa else ""
    content = f"{_MODERATOR_INTRO}{choice}{_FINAL_ONLY}"
    return {"role": "system", "content": _with_prefix(content, prefix)}


def answer_moderator_user(question: str, query_pool, responses: str,
                          prefix: Optional[str] = None) -> dict:
    if prefix is not None:
        return {"role": "user", "content": responses}
    return {"role": "user",
            "content": f"Question: {question}\n{format_query_pool(query_pool)}\n{responses}"}
