import hashlib
from fastapi import Request, Response


def user_cache_key_builder(
    func,
    namespace: str = "",
    request: Request = None,
    response: Response = None,
    *args,
    **kwargs,
):
    """
    Constructs a unique cache key incorporating user UUID, endpoint, and specific query parameters.
    """
    user = kwargs.get("current_user")
    user_uuid = user.user_uuid if user else "anonymous"

    req_url = str(request.url.path)
    if request.url.query:
        req_url += f"?{request.url.query}"

    url_hash = hashlib.md5(req_url.encode()).hexdigest()

    return f"{namespace}:{user_uuid}:{func.__name__}:{url_hash}"
