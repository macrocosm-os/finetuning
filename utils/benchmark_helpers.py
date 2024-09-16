from typing import Callable, Dict, Any, List
import dataclasses


@dataclasses.dataclass
class SubtaskConfig:
    get_score: Callable[[Dict[str, Any]], float]
    lower_bound: float = 0.0


def normalize_within_range(value, lower_bound, higher_bound):
    """Normalizes a benchmark score within a given range.

    Formula as documented in the HF docs.
    """
    return (value - lower_bound) / (higher_bound - lower_bound)


def get_score_from_subtasks(
    results: Dict[str, Any], subtasks: Dict[str, List[SubtaskConfig]]
) -> float:
    """Returns a single score for a leaderboard task group."""
    normalized_scores = []

    for subtask, config in subtasks.items():
        score = config.get_score(results)
        print(f"{subtask} raw score: {score:.2f}")
        if score < config.lower_bound:
            normalized_score = 0
        else:
            normalized_score = (
                normalize_within_range(score, config.lower_bound, 1.0) * 100
            )
        normalized_scores.append(normalized_score)

    return sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0


def compute_ifeval(results: Dict[str, Any]) -> float:
    ifeval = results["leaderboard_ifeval"]
    return get_score_from_subtasks(
        ifeval,
        {
            "prompt_level": SubtaskConfig(
                get_score=lambda x: x["prompt_level_strict_acc,none"], lower_bound=0.0
            ),
            "inst_level": SubtaskConfig(
                get_score=lambda x: x["inst_level_strict_acc,none"], lower_bound=0.0
            ),
        },
    )


def compute_bbh(results: Dict[str, Any]) -> float:
    subtasks = {
        "leaderboard_bbh_boolean_expressions": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_boolean_expressions"][
                "acc_norm,none"
            ],
            lower_bound=0.5,
        ),
        "leaderboard_bbh_causal_judgement": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_causal_judgement"]["acc_norm,none"],
            lower_bound=0.5,
        ),
        "leaderboard_bbh_date_understanding": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_date_understanding"][
                "acc_norm,none"
            ],
            lower_bound=0.16666,
        ),
        "leaderboard_bbh_disambiguation_qa": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_disambiguation_qa"]["acc_norm,none"],
            lower_bound=0.33333,
        ),
        "leaderboard_bbh_formal_fallacies": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_formal_fallacies"]["acc_norm,none"],
            lower_bound=0.5,
        ),
        "leaderboard_bbh_geometric_shapes": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_geometric_shapes"]["acc_norm,none"],
            lower_bound=1 / 11.0,
        ),
        "leaderboard_bbh_hyperbaton": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_hyperbaton"]["acc_norm,none"],
            lower_bound=0.5,
        ),
        "leaderboard_bbh_logical_deduction_five_objects": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_logical_deduction_five_objects"][
                "acc_norm,none"
            ],
            lower_bound=0.2,
        ),
        "leaderboard_bbh_logical_deduction_seven_objects": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_logical_deduction_seven_objects"][
                "acc_norm,none"
            ],
            lower_bound=1 / 7.0,
        ),
        "leaderboard_bbh_logical_deduction_three_objects": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_logical_deduction_three_objects"][
                "acc_norm,none"
            ],
            lower_bound=1 / 3.0,
        ),
        "leaderboard_bbh_movie_recommendation": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_movie_recommendation"][
                "acc_norm,none"
            ],
            lower_bound=1 / 6.0,
        ),
        "leaderboard_bbh_navigate": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_navigate"]["acc_norm,none"],
            lower_bound=0.5,
        ),
        "leaderboard_bbh_object_counting": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_object_counting"]["acc_norm,none"],
            lower_bound=1 / 19.0,
        ),
        "leaderboard_bbh_penguins_in_a_table": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_penguins_in_a_table"][
                "acc_norm,none"
            ],
            lower_bound=0.2,
        ),
        "leaderboard_bbh_reasoning_about_colored_objects": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_reasoning_about_colored_objects"][
                "acc_norm,none"
            ],
            lower_bound=1 / 18.0,
        ),
        "leaderboard_bbh_ruin_names": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_ruin_names"]["acc_norm,none"],
            lower_bound=1 / 6.0,
        ),
        "leaderboard_bbh_salient_translation_error_detection": SubtaskConfig(
            get_score=lambda x: x[
                "leaderboard_bbh_salient_translation_error_detection"
            ]["acc_norm,none"],
            lower_bound=1 / 6.0,
        ),
        "leaderboard_bbh_snarks": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_snarks"]["acc_norm,none"],
            lower_bound=0.5,
        ),
        "leaderboard_bbh_sports_understanding": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_sports_understanding"][
                "acc_norm,none"
            ],
            lower_bound=0.5,
        ),
        "leaderboard_bbh_temporal_sequences": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_temporal_sequences"][
                "acc_norm,none"
            ],
            lower_bound=0.25,
        ),
        "leaderboard_bbh_tracking_shuffled_objects_five_objects": SubtaskConfig(
            get_score=lambda x: x[
                "leaderboard_bbh_tracking_shuffled_objects_five_objects"
            ]["acc_norm,none"],
            lower_bound=0.2,
        ),
        "leaderboard_bbh_tracking_shuffled_objects_seven_objects": SubtaskConfig(
            get_score=lambda x: x[
                "leaderboard_bbh_tracking_shuffled_objects_seven_objects"
            ]["acc_norm,none"],
            lower_bound=1 / 7.0,
        ),
        "leaderboard_bbh_tracking_shuffled_objects_three_objects": SubtaskConfig(
            get_score=lambda x: x[
                "leaderboard_bbh_tracking_shuffled_objects_three_objects"
            ]["acc_norm,none"],
            lower_bound=1 / 3.0,
        ),
        "leaderboard_bbh_web_of_lies": SubtaskConfig(
            get_score=lambda x: x["leaderboard_bbh_web_of_lies"]["acc_norm,none"],
            lower_bound=0.5,
        ),
    }

    return get_score_from_subtasks(results, subtasks)


def compute_math(results: Dict[str, Any]) -> float:
    return get_score_from_subtasks(
        results,
        {
            "leaderboard_math_algebra_hard": SubtaskConfig(
                get_score=lambda x: x["leaderboard_math_algebra_hard"][
                    "exact_match,none"
                ],
                lower_bound=0.0,
            ),
            "leaderboard_math_counting_and_prob_hard": SubtaskConfig(
                get_score=lambda x: x["leaderboard_math_counting_and_prob_hard"][
                    "exact_match,none"
                ],
                lower_bound=0.0,
            ),
            "leaderboard_math_geometry_hard": SubtaskConfig(
                get_score=lambda x: x["leaderboard_math_geometry_hard"][
                    "exact_match,none"
                ],
                lower_bound=0.0,
            ),
            "leaderboard_math_intermediate_algebra_hard": SubtaskConfig(
                get_score=lambda x: x["leaderboard_math_intermediate_algebra_hard"][
                    "exact_match,none"
                ],
                lower_bound=0.0,
            ),
            "leaderboard_math_num_theory_hard": SubtaskConfig(
                get_score=lambda x: x["leaderboard_math_num_theory_hard"][
                    "exact_match,none"
                ],
                lower_bound=0.0,
            ),
            "leaderboard_math_prealgebra_hard": SubtaskConfig(
                get_score=lambda x: x["leaderboard_math_prealgebra_hard"][
                    "exact_match,none"
                ],
                lower_bound=0.0,
            ),
            "leaderboard_math_precalculus_hard": SubtaskConfig(
                get_score=lambda x: x["leaderboard_math_precalculus_hard"][
                    "exact_match,none"
                ],
                lower_bound=0.0,
            ),
        },
    )


def compute_gpqa(results: Dict[str, Any]) -> float:
    return get_score_from_subtasks(
        results,
        {
            "leaderboard_gpqa_diamond": SubtaskConfig(
                get_score=lambda x: x["leaderboard_gpqa_diamond"]["acc_norm,none"],
                lower_bound=0.25,
            ),
            "leaderboard_gpqa_extended": SubtaskConfig(
                get_score=lambda x: x["leaderboard_gpqa_extended"]["acc_norm,none"],
                lower_bound=0.25,
            ),
            "leaderboard_gpqa_main": SubtaskConfig(
                get_score=lambda x: x["leaderboard_gpqa_main"]["acc_norm,none"],
                lower_bound=0.25,
            ),
        },
    )


def compute_musr(results: Dict[str, Any]) -> float:
    return get_score_from_subtasks(
        results,
        {
            "murder_mysteries": SubtaskConfig(
                get_score=lambda x: x["leaderboard_musr_murder_mysteries"][
                    "acc_norm,none"
                ],
                lower_bound=0.5,
            ),
            "object_placement": SubtaskConfig(
                get_score=lambda x: x["leaderboard_musr_object_placements"][
                    "acc_norm,none"
                ],
                lower_bound=0.2,
            ),
            "team_allocation": SubtaskConfig(
                get_score=lambda x: x["leaderboard_musr_team_allocation"][
                    "acc_norm,none"
                ],
                lower_bound=0.3333,
            ),
        },
    )


def compute_mmlu_pro(results: Dict[str, Any]) -> float:
    return get_score_from_subtasks(
        results,
        {
            "leaderboard_mmlu_pro": SubtaskConfig(
                get_score=lambda x: x["leaderboard_mmlu_pro"]["acc,none"],
                lower_bound=0.1,
            ),
        },
    )


def get_leaderboard_scores(results: Dict[str, Any]) -> Dict[str, float]:
    return {
        # "ifeval": compute_ifeval(results),
        "bbh": compute_bbh(results),
        # "math": compute_math(results),
        # "gpqa": compute_gpqa(results),
        # "musr": compute_musr(results),
        "mmlu_pro": compute_mmlu_pro(results),
    }
