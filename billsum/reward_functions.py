"""
Reward functions for GRPO training.

This module contains reward functions for evaluating model-generated summaries:
- FKGL (Flesch-Kincaid Grade Level) for readability assessment
- BERTScore for semantic similarity
- Format-based rewards for completion quality
"""

import textstat
from prompts import parse_summary

MAX_SUMMARY_LENGTH = 512

# ======================
# Metric-based Rewards
# ======================


def fkgl_reward_func(
    prompts,
    completions,
    **kwargs,
) -> list[float]:
    """
    Compute rewards based on Flesch-Kincaid Grade Level (FKGL).

    Evaluates whether the generated summary's readability matches the target
    level specified in the prompt:
    - beginner: FKGL 1-6
    - intermediate: FKGL 6-12
    - advanced: FKGL > 12

    Args:
        prompts: List of conversation prompts containing readability level
        completions: List of model-generated completions
        **kwargs: Additional keyword arguments

    Returns:
        list[float]: Reward scores based on FKGL matching the target range
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    # Define FKGL ranges for each readability level
    FKGL_RANGES = {
        "beginner": (0.0, 6.0),
        "intermediate": (6.0, 12.0),
        "advanced": (12.0, float("inf")),
    }

    for i, response in enumerate(responses):
        # Parse content between <summary> tags
        text = parse_summary(response)

        if len(text) == 0:
            rewards.append(-1.0)
            continue

        try:
            # Extract readability level from prompt
            prompt_content = prompts[i][-1]["content"]
            target_level = None

            for level in ["beginner", "intermediate", "advanced"]:
                if level in prompt_content.lower():
                    target_level = level
                    break

            if target_level is None:
                # If no level found, give neutral reward
                rewards.append(0.0)
                continue

            # Calculate FKGL score on parsed summary content
            fkgl_score = textstat.flesch_kincaid_grade(text)

            # Get expected range for this level
            min_fkgl, max_fkgl = FKGL_RANGES[target_level]

            # Check if FKGL is within the expected range
            if min_fkgl <= fkgl_score <= max_fkgl:
                # Perfect match: within range
                # Calculate how centered it is within the range (optional refinement)
                if max_fkgl == float("inf"):
                    # For advanced, reward being above 12
                    reward = min(1.0, 0.5 + (fkgl_score - 12.0) / 10.0)
                else:
                    # For beginner/intermediate, reward being in range
                    range_size = max_fkgl - min_fkgl
                    center = (min_fkgl + max_fkgl) / 2
                    distance_from_center = abs(fkgl_score - center)
                    reward = max(0.5, 1.0 - (distance_from_center / range_size))
                rewards.append(reward)
            else:
                # Out of range: negative penalty based on distance
                if fkgl_score < min_fkgl:
                    distance = min_fkgl - fkgl_score
                else:
                    distance = fkgl_score - max_fkgl

                # Negative penalty increases with distance
                # Distance of 0 (just at boundary) → -0.1
                # Distance of 5 → -0.5
                # Distance of 10+ → -1.0
                penalty = max(-1.0, -0.1 - (distance / 10.0))
                rewards.append(penalty)

        except Exception:
            # If FKGL calculation fails (e.g., too short text)
            rewards.append(0.0)

    return rewards


def bertscore_reward_func(
    prompts,
    completions,
    answer,
    **kwargs,
) -> list[float]:
    """
    Compute rewards based on BERTScore F1.

    Evaluates semantic similarity between generated summaries and reference
    answers using BERTScore. Parses content between <summary> tags before scoring.

    Args:
        prompts: List of conversation prompts
        completions: List of model-generated completions
        answer: Reference answers for comparison
        **kwargs: Additional keyword arguments

    Returns:
        list[float]: BERTScore F1 scores for each completion
    """
    # Parse content between <summary> tags for each response
    responses = [parse_summary(completion[0]["content"]) for completion in completions]

    try:
        from bert_score import score

        references = (
            [answer] * len(responses)
            if isinstance(answer, str)
            else answer * len(responses)
        )
        _, _, f1_flat = score(
            responses,
            references,
            model_type="roberta-large",
            lang="en",
            batch_size=16,
            device="cuda:0",
            verbose=False,
        )
        return f1_flat.tolist()

    except ImportError:
        raise ImportError(
            "bert_score package required. Install with: pip install bert-score"
        )


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Compute format-based rewards for generated completions.

    Evaluates completion quality based on:
    - Proper <summary></summary> tags
    - Non-empty responses

    Args:
        completions: List of model-generated completions
        **kwargs: Additional keyword arguments

    Returns:
        list[float]: Format quality scores for each completion
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        score = 0.0

        # 1. Clean response for checking
        text = response.strip()

        if len(text) == 0:
            rewards.append(-1.0)  # Heavy penalty for empty responses
            continue

        # 2. Check for proper summary tags
        has_opening_tag = "<summary>" in text
        has_closing_tag = "</summary>" in text

        if has_opening_tag and has_closing_tag:
            # Both tags present - good completion
            score += 1.0
        elif has_opening_tag and not has_closing_tag:
            # Started but didn't finish - penalty for incomplete
            score -= 0.5
        elif not has_opening_tag:
            # No opening tag - format violation
            score -= 1.0

        rewards.append(score)

    return rewards
