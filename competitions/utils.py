from typing import Optional

import constants
from competitions.data import Competition, CompetitionId


def get_competition(id: CompetitionId) -> Optional[Competition]:
    """Returns the competition with the given id, or None if it does not exist."""
    for x in constants.COMPETITION_SCHEDULE:
        if x.id == id:
            return x
    return None
