"""Project name and the identifiers derived from it.

Rename the project here first. The import package directory (``postprism/``) and
the docs still need a search-and-replace; see docs/RENAMING.md.
"""

SERVICE_NAME = "PostPrism"
ENV_PREFIX = "POSTPRISM_"
RUNTIME_PROFILE_ENV = f"{ENV_PREFIX}RUNTIME_PROFILE"
# Build manifest written into the locally built RAM++ wheel and checked at startup.
RAM_BUILD_MANIFEST = "postprism_ram_build.json"
