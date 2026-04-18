from src.generation.prompt import format_prompt


def test_format_prompt_includes_context_and_question() -> None:
    result = format_prompt(context="ctx", question="q")
    assert "Context:" in result
    assert "ctx" in result
    assert "Question: q" in result
