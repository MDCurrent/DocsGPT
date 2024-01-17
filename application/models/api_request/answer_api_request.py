from application.core.settings import settings
from application.models.vectorstore.base import get_vectorstore
from pydantic import BaseModel, Extra
from typing import Optional

class AnswerApiRequest(BaseModel):
    config = {"arbitrary_settings": False}
    extra = Extra.forbid

    question: str
    history: list[dict]
    conversation_id: str
    prompt_id: str = "default"
    api_key: str = settings.API_KEY
    embeddings_key: Optional[str] = None

    @property
    def active_docs(self) -> Optional[str]:
        """Return `active_docs` value."""
        return self.get("active_docs")

    def initialize_model(self, data: dict):
        self.question = data["question"]
        self.history = data["history"]
        self.conversation_id = data["conversation_id"]
        self.prompt_id = data.get("prompt_id", self.prompt_id)
        self._initialize_keys(data)
        self._initialize_vectorstore()

    def _initialize_keys(self, data):
        self.api_key = data.get("api_key", self.api_key)
        self.embeddings_key = data.get("embeddings_key", self.embeddings_key)

    def _initialize_vectorstore(self):
        if self.active_docs is not None:
            self.vectorstore = get_vectorstore({"active_docs": self.active_docs})