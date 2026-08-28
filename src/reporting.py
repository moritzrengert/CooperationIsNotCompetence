def common_row(context, player, decision):
    reasoning_text = decision.thinking if str(decision.thinking or "").strip() else decision.reason
    reasoning_source = "thinking" if str(decision.thinking or "").strip() else ("reason" if str(decision.reason or "").strip() else "")
    row = {
        **context["meta"],
        "treatment": context["treatment"],
        "chain_id": context["chain"],
        "generation": context["generation"],
        "player": player,
        "reason": decision.reason,
        "thinking": decision.thinking,
        "reasoning_text": reasoning_text,
        "reasoning_text_source": reasoning_source,
        "raw_response": decision.raw_response,
        "fallback_used": decision.fallback_used,
        "fallback_error": decision.fallback_error,
        "parser_mode": decision.parser_mode,
        "retry_count": decision.retry_count,
        "repetition": context["repetition"],
        "effective_seed": context["effective_seed"](
            context["repetition"],
            context["chain"],
            context["generation"],
            player,
        ),
    }
    row.update(decision.api_usage)
    row.update(decision.component_log)
    return row
