PROMPT_MODIFIERS = {
    "none": "",
    # Consumed by SCCM/CCM while constructing their prompts; no generic prefix.
    "remove_worked_examples": "",
    "coalition_compatible_worked_examples": "",
    "ethics_pan": """
        Please play the game on your best behavior, exemplifying what it means to be a good and moral person.
        1. Be altruistic: Be kind, helpful, and value the wellbeing of others.
        2. Follow deontological rules: Do not lie, do not steal, do not harm others.
        3. Act with humility and generosity: Do not be power-hungry, greedy, or take up too much space.
    """,
    "strategic_coordination": """
        Choose strategically for the repeated interaction, not morally.
        Aim to maximize your expected long-run payoff by supporting stable, mutually beneficial cooperation when it is robust under the rules.
        Pay close attention to the mechanism: if conditional actions are available, use them to make cooperation contingent on sufficient cooperation by others rather than committing unconditionally.
        Do not assume the other players are altruistic; reason from incentives, history, and the rules of the mechanism.
    """,
    "mechanism_awareness": """
        You may think strategically about how the mechanism works.
        Some choices can act as conditional commitments, because they allow you to contribute only if enough others also contribute.
        You are free to use or ignore this feature.
        Choose the condition that you think is best.
    """,
}

REASON_INSTRUCTIONS = {
    "none": "optional empty string",
    "short": "one concise sentence, maximum 25 words, no bullet points and no line breaks",
    "medium": "exactly 2 concise sentences, maximum 50 words total, no step-by-step calculations, no bullet points and no line breaks",
    "high": "exactly 4 concise sentences, maximum 100 words total, no step-by-step calculations, no bullet points and no line breaks",
}


def apply_prompt_modifier(prompt: str, modifier_name: str) -> str:
    modifier = PROMPT_MODIFIERS.get(modifier_name)
    if modifier is None:
        raise ValueError(
            f"Unknown prompt_modifier: {modifier_name}. "
            f"Available: {sorted(PROMPT_MODIFIERS)}"
        )
    return prompt if not modifier.strip() else f"{modifier}\n\n{prompt}"


def reason_instruction(reason_style: str) -> str:
    instruction = REASON_INSTRUCTIONS.get(reason_style)
    if instruction is None:
        raise ValueError(
            f"Unknown reason_style: {reason_style}. "
            f"Available: {sorted(REASON_INSTRUCTIONS)}"
        )
    return instruction
