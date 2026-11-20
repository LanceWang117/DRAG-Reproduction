"""Every baseline prompt string, verbatim, as small pure builders.

Text must not drift: the golden transcript (T0) compares the rendered prompts with the
frozen ``model/drag.py`` character for character. The StrategyQA branches are kept as the
baseline has them, including the missing space before ``Question:`` in the debate message.
"""
from __future__ import annotations

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


def format_reference(retrieval_result) -> str:
    out = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        out += f"Doc {idx+1}(Title: {title}) {text}\n"
    return out


def format_query_pool(query_pool) -> str:
    out = ""
    for i, query in enumerate(query_pool):
        out += f"Query {i+1}: {query}\nRetrieved Content: {format_reference(query_pool[query])}"
    return out


def query_agent_system(agent_name: str) -> dict:
    if "Proponent" in agent_name:
        return {"role": "system", "content": PROPONENT_QUERY}
    if "Opponent" in agent_name:
        return {"role": "system", "content": OPPONENT_QUERY}
    raise ValueError("The agent name must contain either Proponent or Opponent")


def query_round_user(question: str, query_pool) -> dict:
    return {"role": "user", "content": f"Question: {question}\n{format_query_pool(query_pool)}"}


def query_moderator_system(agent_outputs: dict, input_query: str, query_pool) -> dict:
    """``agent_outputs`` maps agent name to its raw output for the current round."""
    arguments = "".join(f"{agent}: {output}\n" for agent, output in agent_outputs.items())
    return {"role": "system", "content": (
        "You are the judge in a debate. Your task is to evaluate the arguments from agents. "
        "There are two types of agents:\n"
        "1. Proponent Agent: Argues that the current retrieved content is sufficient.\n"
        "2. Opponent Agent: Argues that the current retrieved content is insufficient and "
        "proposes query refinement.\n"
        f"Question: {input_query}\n{format_query_pool(query_pool)}\n"
        f"Agents Arguments:\n{arguments}\nOutput only the agent's name. No explanations.")}


def judge_retry_user() -> dict:
    return {"role": "user", "content": JUDGE_RETRY}


def question_user(question: str) -> dict:
    return {"role": "user", "content": f"Question: {question}\n"}


def answer_agent_system(agent_name: str, query_pool, strategy_qa: bool) -> dict:
    fmt = _ANSWER_FORMAT_YES_NO if strategy_qa else _ANSWER_FORMAT
    if "Proponent" in agent_name:
        return {"role": "system", "content": (
            f"Answer the question based on the given document. {fmt} "
            f"The following are given documents.\n{format_query_pool(query_pool)}")}
    return {"role": "system",
            "content": f"Answer the question based on your own knowledge. {fmt}"}


def answer_only_system(query_pool, strategy_qa: bool) -> dict:
    choice = _YES_NO_CHOICE if strategy_qa else ""
    return {"role": "system", "content": (
        f"Answer the question based on the given document. {choice}{_FINAL_ONLY}"
        f"\n{format_query_pool(query_pool)}")}


def answer_debate_user(others: dict, question: str, round_index: int,
                       strategy_qa: bool) -> dict:
    """``others`` maps agent name to its chat history; round r reads index 2r."""
    fmt = _ANSWER_FORMAT_YES_NO if strategy_qa else _ANSWER_FORMAT
    text = f"{_DEBATE_INTRO}{fmt}Question: {question}\nOther agents responses:\n"
    for agent_name, history in others.items():
        text += f"{agent_name}: {history[round_index*2]['content']}\n"
    return {"role": "user", "content": text}


def answer_moderator_system(strategy_qa: bool) -> dict:
    choice = _YES_NO_CHOICE if strategy_qa else ""
    return {"role": "system", "content": f"{_MODERATOR_INTRO}{choice}{_FINAL_ONLY}"}


def answer_moderator_user(question: str, query_pool, responses: str) -> dict:
    return {"role": "user",
            "content": f"Question: {question}\n{format_query_pool(query_pool)}\n{responses}"}
