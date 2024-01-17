from pydantic import BaseModel, Field

class UploadFileInfo(BaseModel):
    user: str = Field(..., alias="user")
    name: str = Field(..., alias="name")
    file: str = Field(..., alias="file")

    @property
    def final_save_path(self):
        save_dir = os.path.join(current_dir, settings.UPLOAD_FOLDER, self.user, self.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return os.path.join(save_dir, self.file.filename)