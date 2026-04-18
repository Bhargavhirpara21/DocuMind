from __future__ import annotations

QA_PROMPT_TEMPLATE = """You are a manufacturing engineering assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I cannot find this information in the available documents."

Rules:
- Be precise and technical
- Include specific numbers, values, and specifications when available
- Always cite the source document and page number
- If the context is in German, you may answer in German or English based on the question language

Context:
{context}

Question: {question}

Answer:
"""


def format_prompt(context: str, question: str) -> str:
    return QA_PROMPT_TEMPLATE.format(context=context, question=question)
