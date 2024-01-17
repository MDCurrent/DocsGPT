from pydantic import BaseModel, FilePath
from typing import List
from pathlib import Path
import tempfile

class UploadApiRequest(BaseModel):
    user: str
    name: str
    language: str = Field(..., alias="job_name")
    file_faiss: IO[bytes] = Field(..., title="FaISS Index File", alias="file_faiss")
    file_pkl: IO[bytes] = Field(..., title="Pickle Data File", alias="file_pkl")

    @property
    def location(self) -> Path:
        save_dir = Path(os.path.join(os.getcwd(), "indexes", self.user, self.name))
        if not save_dir.parent.is_dir():
            save_dir.parent.mkdir(parents=True, exist_ok=True)
        return save_dir / f"index_{self.name}"

    def save_files(self):
        self.file_faiss.seek(0)
        self.file_pkl.seek(0)
        self.file_faiss.save(self.location / "index.faiss")
        self.file_pkl.save(self.location / "index.pkl")

    def insert_into_vectors_collection(self):
        import datetime

        collection = vectors_collection

        inserted_id = collection.insert_one(
            {
                "user": self.user,
                "name": self.name,
                "language": self.language,
                "location": str(self.location),
                "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "model": settings.EMBEDDINGS_NAME,
                "type": "local",
                "_id": str(ObjectId()),
            }
        ).inserted_id

        return inserted_id