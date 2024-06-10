import abc

from constants import Competition
from model.data import Model, ModelId


class RemoteModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model."""

    @abc.abstractmethod
    async def upload_model(
        self, model: Model, competition: Competition
    ) -> ModelId:
        """Uploads a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    async def download_model(
        self, model_id: ModelId, local_path: str, competition: Competition
    ) -> Model:
        """Retrieves a trained model from the appropriate location and stores at the given path."""
        pass
