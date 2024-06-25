from typing import List, Optional

import constants
from competitions.data import Competition, CompetitionId, ModelConstraints


def get_model_constraints(id: CompetitionId) -> Optional[ModelConstraints]:
    """Returns the model constraints for the given id, or None if it does not exist."""
    return constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(id, None)


def get_competition_for_block(id: CompetitionId, block: int) -> Optional[Competition]:
    """Returns the competition for the given id at the given block, or None if it does not exist."""
    competition_schedule = get_competition_schedule_for_block(block)
    for comp in competition_schedule:
        if comp.id == id:
            return comp
    return None


def get_competition_schedule_for_block(block: int) -> List[Competition]:
    """Returns the competition schedule at block."""
    competition_schedule = None
    for b, schedule in constants.COMPETITION_SCHEDULE_BY_BLOCK:
        if block >= b:
            competition_schedule = schedule
    assert (
        competition_schedule is not None
    ), f"No competition schedule found for block {block}"
    return competition_schedule
