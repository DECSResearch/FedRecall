import os
from typing import Optional


def _in_colab() -> bool:
    try:
        # Colab presence heuristic
        return os.path.exists("/content")
    except Exception:
        return False


def resolve_output_path(path: Optional[str]) -> Optional[str]:
    """
    Resolve an output path to MyDrive when running in Google Colab.
    Rules:
      - If not in Colab or path is falsy, return as-is.
      - If absolute and already under /content/drive/, keep as-is.
      - If absolute under /content/, remap /content/... -> /content/drive/MyDrive/...
      - If relative, prefix with /content/drive/MyDrive/
    """
    if not path or not _in_colab():
        return path

    base = "/content/drive/MyDrive"
    try:
        if os.path.isabs(path):
            norm = os.path.normpath(path)
            if norm.startswith("/content/drive/"):
                return norm
            if norm.startswith("/content/"):
                # Map /content/... to MyDrive/...
                rel = os.path.relpath(norm, "/content")
                return os.path.normpath(os.path.join(base, rel))
            # Other absolute paths: leave unchanged
            return norm
        # Relative path -> under MyDrive
        return os.path.normpath(os.path.join(base, path))
    except Exception:
        return path


