import dataclasses
from typing import ClassVar, Optional, Type

from transformers import PreTrainedModel

from competitions.data import CompetitionId

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128
# The length, in bytes, of a git commit hash.
GIT_COMMIT_LENGTH = 40
# The length, in bytes, of a base64 encoded sha256 hash.
SHA256_BASE_64_LENGTH = 44
# The max length, in characters, of the competition id
MAX_COMPETITION_ID_LENGTH = 2


@dataclasses.dataclass
class ModelId:
    """Uniquely identifies a trained model"""

    MAX_REPO_ID_LENGTH: ClassVar[int] = (
        MAX_METADATA_BYTES
        - GIT_COMMIT_LENGTH
        - SHA256_BASE_64_LENGTH
        - MAX_COMPETITION_ID_LENGTH
        - 4  # separators
    )

    # Namespace where the model can be found. ex. Hugging Face username/org.
    namespace: str

    # Name of the model.
    name: str

    # Identifier for competition
    competition_id: CompetitionId

    # When handling a model locally the commit and hash are not necessary.
    # Commit must be filled when trying to download from a remote store.
    commit: Optional[str] = dataclasses.field(default=None)

    # Hash is filled automatically when uploading to or downloading from a remote store.
    hash: Optional[str] = dataclasses.field(default=None)

    # The secure hash that's used for validation.
    secure_hash: Optional[str] = dataclasses.field(default=None)

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.namespace}:{self.name}:{self.commit}:{self.secure_hash}:{self.competition_id.value}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ModelId"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        competition_id = int(tokens[4])
        assert competition_id in [c.value for c in CompetitionId]

        return cls(
            namespace=tokens[0],
            name=tokens[1],
            commit=tokens[2] if tokens[2] != "None" else None,
            secure_hash=tokens[3] if tokens[3] != "None" else None,
            competition_id=competition_id,
        )


@dataclasses.dataclass
class Model:
    """Represents a pre trained foundation model."""

    class Config:
        arbitrary_types_allowed = True

    # Identifier for this model.
    id: ModelId

    # PreTrainedModel.base_model returns torch.nn.Module if needed.
    pt_model: PreTrainedModel


@dataclasses.dataclass
class ModelMetadata:
    # Identifier for this trained model.
    id: ModelId

    # Block on which this model was uploaded on the chain.
    block: int
