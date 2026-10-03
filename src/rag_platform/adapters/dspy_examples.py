"""Curated examples for the DSPy programs (specification section 10, "Curated examples").

These are written and reviewed by hand, not generated: the specification rules out synthetic data
as the only evaluation source. They are deliberately small — enough to bootstrap demonstrations
and to hold some back — and each list keeps its classes balanced so the held-out split, which
takes every third example, still sees every class.
"""

import importlib
from typing import Any

importlib.import_module("numpy")  # see dspy_programs: numpy must be fully loaded before DSPy

import dspy  # noqa: E402


def _examples(rows: list[dict[str, Any]], *inputs: str) -> list[Any]:
    return [dspy.Example(**row).with_inputs(*inputs) for row in rows]


def classifier_examples() -> list[Any]:
    return _examples(
        [
            {"question": "How do I request annual leave?", "intent": "procedural"},
            {"question": "Who approves an expense claim?", "intent": "factual"},
            {"question": "leave", "intent": "ambiguous"},
            {"question": "What are the steps to submit an expense claim?", "intent": "procedural"},
            {"question": "What is the notice period for annual leave?", "intent": "factual"},
            {
                "question": "What is the difference between annual leave and sick leave?",
                "intent": "comparison",
            },
            {"question": "How do I reset my portal password?", "intent": "procedural"},
            {"question": "Compare the travel and expense policies.", "intent": "comparison"},
            {"question": "policy?", "intent": "ambiguous"},
            {"question": "When are expense claims paid?", "intent": "factual"},
            {
                "question": "Annual leave versus unpaid leave: which applies?",
                "intent": "comparison",
            },
            {"question": "it", "intent": "ambiguous"},
        ],
        "question",
    )


def rewriter_examples() -> list[Any]:
    return _examples(
        [
            {
                "question": "How do I book vacation?",
                "intent": "procedural",
                "rewritten": "How do I request annual leave?",
                "expected_terms": ["leave"],
            },
            {
                "question": "What is the PTO policy?",
                "intent": "factual",
                "rewritten": "What is the annual leave policy in the handbook?",
                "expected_terms": ["leave"],
            },
            {
                "question": "Who signs off my expenses?",
                "intent": "factual",
                "rewritten": "Who approves an expense claim?",
                "expected_terms": ["expense"],
            },
            {
                "question": "How do I get holidays approved?",
                "intent": "procedural",
                "rewritten": "How is an annual leave request approved?",
                "expected_terms": ["leave"],
            },
            {
                "question": "Where do I file a reimbursement?",
                "intent": "procedural",
                "rewritten": "Where do I submit an expense claim?",
                "expected_terms": ["expense", "claim"],
            },
            {
                "question": "How much time off do I get?",
                "intent": "factual",
                "rewritten": "How much annual leave am I entitled to?",
                "expected_terms": ["leave"],
            },
        ],
        "question",
        "intent",
    )


def decomposer_examples() -> list[Any]:
    return _examples(
        [
            {
                "question": "How do I request annual leave and who approves expense claims?",
                "max_subqueries": 3,
                "subqueries": ["How do I request annual leave?", "Who approves expense claims?"],
                "expected_count": 2,
            },
            {
                "question": "How do I request annual leave?",
                "max_subqueries": 3,
                "subqueries": ["How do I request annual leave?"],
                "expected_count": 1,
            },
            {
                "question": "When are claims paid; who approves them; where are they submitted?",
                "max_subqueries": 3,
                "subqueries": [
                    "When are expense claims paid?",
                    "Who approves expense claims?",
                    "Where are expense claims submitted?",
                ],
                "expected_count": 3,
            },
            {
                "question": "What is the notice period for leave and for resignation?",
                "max_subqueries": 3,
                "subqueries": [
                    "What is the notice period for annual leave?",
                    "What is the notice period for resignation?",
                ],
                "expected_count": 2,
            },
            {
                "question": "Who owns the cost centre?",
                "max_subqueries": 3,
                "subqueries": ["Who owns the cost centre?"],
                "expected_count": 1,
            },
            {
                "question": "How is leave requested and how is it approved?",
                "max_subqueries": 3,
                "subqueries": ["How is annual leave requested?", "How is annual leave approved?"],
                "expected_count": 2,
            },
        ],
        "question",
        "max_subqueries",
    )


LEAVE = (
    "[C1] Annual leave requests must be submitted in the HR portal at least five working days "
    "before the first day of leave. Managers approve or decline a request within two working days."
)
EXPENSES = (
    "[C2] Expense claims are submitted monthly and are approved by the owner of the cost centre "
    "that funds the claim."
)


def synthesizer_examples() -> list[Any]:
    return _examples(
        [
            {
                "question": "How do I request annual leave?",
                "evidence": [LEAVE, EXPENSES],
                "max_sentences": 2,
                "answer": "Submit the request in the HR portal at least five working days ahead.",
                "citation_ids": ["C1"],
                "expected_citation_ids": ["C1"],
                "expected_terms": ["portal"],
            },
            {
                "question": "Who approves an expense claim?",
                "evidence": [LEAVE, EXPENSES],
                "max_sentences": 1,
                "answer": "The owner of the cost centre that funds the claim approves it.",
                "citation_ids": ["C2"],
                "expected_citation_ids": ["C2"],
                "expected_terms": ["cost centre"],
            },
            {
                "question": "When is the next lunar eclipse?",
                "evidence": [LEAVE, EXPENSES],
                "max_sentences": 2,
                "answer": "",
                "citation_ids": [],
                "expected_citation_ids": [],
                "expected_terms": [],
            },
            {
                "question": "How quickly is a leave request decided?",
                "evidence": [LEAVE],
                "max_sentences": 1,
                "answer": "Managers approve or decline a request within two working days.",
                "citation_ids": ["C1"],
                "expected_citation_ids": ["C1"],
                "expected_terms": ["two working days"],
            },
            {
                "question": "How often are expense claims submitted?",
                "evidence": [EXPENSES],
                "max_sentences": 1,
                "answer": "Expense claims are submitted monthly.",
                "citation_ids": ["C2"],
                "expected_citation_ids": ["C2"],
                "expected_terms": ["monthly"],
            },
            {
                "question": "What is the company's revenue target?",
                "evidence": [LEAVE],
                "max_sentences": 2,
                "answer": "",
                "citation_ids": [],
                "expected_citation_ids": [],
                "expected_terms": [],
            },
        ],
        "question",
        "evidence",
        "max_sentences",
    )


def verifier_examples() -> list[Any]:
    return _examples(
        [
            {
                "answer": "Submit the request in the HR portal at least five working days ahead.",
                "evidence": [LEAVE],
                "unsupported_claims": [],
                "grounded": True,
            },
            {
                "answer": "Annual leave is unlimited.",
                "evidence": [LEAVE],
                "unsupported_claims": ["Annual leave is unlimited."],
                "grounded": False,
            },
            {
                "answer": "The owner of the cost centre approves the claim.",
                "evidence": [EXPENSES],
                "unsupported_claims": [],
                "grounded": True,
            },
            {
                "answer": "Managers decide within two working days. Requests can be made by phone.",
                "evidence": [LEAVE],
                "unsupported_claims": ["Requests can be made by phone."],
                "grounded": False,
            },
            {
                "answer": "Expense claims are submitted monthly.",
                "evidence": [EXPENSES],
                "unsupported_claims": [],
                "grounded": True,
            },
            {
                "answer": "Expense claims are approved by the finance director.",
                "evidence": [EXPENSES],
                "unsupported_claims": ["Expense claims are approved by the finance director."],
                "grounded": False,
            },
        ],
        "answer",
        "evidence",
    )


def clarifier_examples() -> list[Any]:
    return _examples(
        [
            {
                "question": "leave",
                "clarification": "Do you mean requesting annual leave, or another kind of leave?",
            },
            {
                "question": "policy?",
                "clarification": "Which policy are you asking about, and what do you need from it?",
            },
            {
                "question": "it",
                "clarification": "What would you like to know, and about which document or team?",
            },
            {
                "question": "expenses",
                "clarification": "Are you asking how to submit a claim, or who approves it?",
            },
            {
                "question": "approval",
                "clarification": "Which request needs approval: annual leave or an expense claim?",
            },
            {
                "question": "portal",
                "clarification": "What are you trying to do in the HR portal?",
            },
        ],
        "question",
    )
