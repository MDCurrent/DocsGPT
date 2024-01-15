from __future__ import annotations

from typing import TypeVar, Dict, Any
from pydantic import ValidationError

T = TypeVar('T')

def dict_to_validated_model(model: T, data: Dict[str, Any]) -> T:
    try:
        result = model.parse_obj(data)
    except ValidationError as err:
        raise ValueError(err.errors()[0]['msg'])
    else:
        return result