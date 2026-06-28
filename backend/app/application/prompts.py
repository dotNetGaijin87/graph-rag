"""Prompt templates for extraction and answering."""

from __future__ import annotations

from ..domain.models import RetrievalContext

EXTRACTION_SYSTEM = (
    "You are an expert at building knowledge graphs from text. "
    "You extract the key entities and the relationships between them. "
    "Respond with strict JSON only — no prose, no markdown."
)

EXTRACTION_PROMPT_TEMPLATE = """\
Extract a knowledge graph from the text below.

Return JSON with exactly this shape:
{{
  "entities": [
    {{"name": "<canonical name>", "type": "<Person|Organization|Location|Concept|Product|Event|Other>", "description": "<short description>"}}
  ],
  "relationships": [
    {{"source": "<entity name>", "target": "<entity name>", "type": "<UPPER_SNAKE_CASE verb>", "description": "<short description>"}}
  ]
}}

Rules:
- Use concise, canonical entity names (e.g. "Marie Curie", not "she").
- Every relationship's source and target MUST appear in the entities list.
- Prefer 5-20 of the most important entities; do not invent facts.
- If nothing meaningful can be extracted, return empty lists.

Example
Text:
\"\"\"
Ada Lovelace wrote the first algorithm for the Analytical Engine, a mechanical
computer designed by Charles Babbage.
\"\"\"
JSON:
{{"entities": [{{"name": "Ada Lovelace", "type": "Person", "description": "Wrote the first algorithm"}}, {{"name": "Analytical Engine", "type": "Product", "description": "Mechanical computer"}}, {{"name": "Charles Babbage", "type": "Person", "description": "Designed the Analytical Engine"}}], "relationships": [{{"source": "Ada Lovelace", "target": "Analytical Engine", "type": "WROTE_ALGORITHM_FOR", "description": ""}}, {{"source": "Charles Babbage", "target": "Analytical Engine", "type": "DESIGNED", "description": ""}}]}}

Now extract from this text:
\"\"\"
{text}
\"\"\"
"""


def extraction_prompt(text: str) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(text=text)


ANSWER_SYSTEM = (
    "You answer questions using ONLY the information provided to you, which comes from "
    "the user's own knowledge base. Write the answer in a natural, direct voice, as if "
    "the knowledge were your own. Do NOT mention or cite 'the context', 'the passages', "
    "'sources', 'background information', or any passage numbers, and do not say things "
    "like 'based on the provided text'. If the information given does not answer the "
    "question, say you don't have enough information. Be concise and factual, and never "
    "invent details."
)

ANSWER_PROMPT_TEMPLATE = """\
Using only the information below, answer the question in a natural voice. Do not refer to
this information as context, passages, or sources, and do not number or cite it.

# Known facts
{facts}

# Reference notes
{passages}

# Question
{question}

Answer using only the information above. If it does not contain the answer, reply exactly:
"I don't have enough information to answer that."

# Answer
"""


def answer_prompt(question: str, context: RetrievalContext) -> str:
    if context.facts:
        facts = "\n".join(f"- {fact.as_sentence()}" for fact in context.facts)
    else:
        facts = "(none)"

    if context.chunks:
        passages = "\n\n".join(chunk.text for chunk in context.chunks)
    else:
        passages = "(none)"

    return ANSWER_PROMPT_TEMPLATE.format(facts=facts, passages=passages, question=question)


SUMMARIZE_SYSTEM = (
    "You merge several descriptions of the same entity or relationship into one "
    "comprehensive description. Include the information from every description, resolve "
    "any contradictions, remove redundancy, write in the third person, and name the "
    "subject so the result stands on its own. Respond with the description only."
)

SUMMARIZE_PROMPT_TEMPLATE = """\
Subject: {subject}

Descriptions:
{descriptions}

Merged description:
"""


def summarize_prompt(subject: str, descriptions: list[str]) -> str:
    listed = "\n".join(f"- {description}" for description in descriptions)
    return SUMMARIZE_PROMPT_TEMPLATE.format(subject=subject, descriptions=listed)
