from __future__ import annotations

import os
import json


def main() -> None:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "(gcloud auth)")
    loc = os.environ.get("VERTEX_LOCATION", "us-central1")
    ok = bool(project)
    print(json.dumps({
        "project": project,
        "location": loc,
        "credentials": creds,
        "ok": ok
    }, indent=2))


if __name__ == "__main__":
    main()


