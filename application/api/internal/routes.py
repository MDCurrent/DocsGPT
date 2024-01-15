from http import HTTPStatus
import os
import datetime
from application.models.api_request.upload_api_request import UploadApiRequest
from flask import Blueprint, request, send_from_directory
from pymongo import MongoClient
from werkzeug.utils import secure_filename


from application.core.settings import settings
from application.models.api_request.upload_api_request import UploadApiRequest
mongo = MongoClient(settings.MONGO_URI)
db = mongo["docsgpt"]
conversations_collection = db["conversations"]
vectors_collection = db["vectors"]

current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


internal = Blueprint('internal', __name__)
@internal.route("/api/download", methods=["get"])
def download_file():
    user = secure_filename(request.args.get("user"))
    job_name = secure_filename(request.args.get("name"))
    filename = secure_filename(request.args.get("file"))
    save_dir = os.path.join(current_dir, settings.UPLOAD_FOLDER, user, job_name)
    return send_from_directory(save_dir, filename, as_attachment=True)



@internal.route("/api/upload_index", methods=["POST"])
def upload_index_files():
    """Upload two files(index.faiss, index.pkl) to the user's folder."""
    try:
        # Rest of the code...

        form_data = UploadApiRequest(user=user, name=name, language=lang, file_faiss=io.BytesIO(file_faiss.read()), file_pkl=io.BytesIO(file_pkl.read()))

        # Rest of the code...
    except KeyError as exc:
        return {'message': f'Missing field "{exc}".'}, HTTPStatus.BAD_REQUEST
    except Exception as exc:
        return {'message': f'Internal Server Error: {exc}'}, HTTPStatus.INTERNAL_SERVER_ERROR