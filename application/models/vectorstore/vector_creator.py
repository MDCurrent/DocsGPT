from application.core.settings import settings
from application.models.vectorstore.faiss import FaissStore
from application.models.vectorstore.elasticsearch import ElasticsearchStore
from application.models.vectorstore.mongodb import MongoDBVectorStore


class VectorCreator:
    vectorstores = {
        'faiss': FaissStore,
        'elasticsearch':ElasticsearchStore,
        'mongodb': MongoDBVectorStore,
    }

    @classmethod
    def create_vectorstore(cls, type=settings.VECTOR_STORE, *args, **kwargs):
        vectorstore_class = cls.vectorstores.get(type.lower())
        if not vectorstore_class:
            raise ValueError(f"No vectorstore class found for type {type}")
        return vectorstore_class(*args, **kwargs)