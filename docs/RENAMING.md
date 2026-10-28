# Renaming the project

The working name is PostPrism. It appears in these places:

| Where | What |
| --- | --- |
| `postprism/branding.py` | `SERVICE_NAME` (FastAPI title), `ENV_PREFIX` (`POSTPRISM_`), `RUNTIME_PROFILE_ENV`, and `RAM_BUILD_MANIFEST` (`postprism_ram_build.json`, written into the locally built RAM++ wheel) |
| `postprism/` | the import package directory, and every `import postprism` / `from postprism` line |
| `postprism/api/application.py` | `SERVICE_LOGGERS = ("app", "postprism")`, the logger names that get the service log format |
| `scripts/build_ram_package.py` | the temporary build directory prefix `postprism-ram-build-` |
| `tests/` | literal `POSTPRISM_RUNTIME_PROFILE`, `POSTPRISM_ARTIFACTS`, and `postprism_ram_build.json` strings, so the tests catch an unplanned rename |
| `app.py`, `postprism/__init__.py` | docstrings |
| `README.md`, `AGENTS.md`, `NOTICE`, `docs/` | prose, and the `POSTPRISM_ARTIFACTS` install variable |

`pyproject.toml` has no project name; it only configures ruff and pytest.

## Recipe

Replace `newname` / `NewName` / `NEWNAME_` with the new name, from the repository
root, on a clean working tree:

```bash
git mv postprism newname
git grep -lz -e postprism -e PostPrism -e POSTPRISM | xargs -0 sed -i \
  -e 's/postprism/newname/g' -e 's/PostPrism/NewName/g' -e 's/POSTPRISM_/NEWNAME_/g'
./verify.sh
git grep -niE 'postprism'   # should print nothing
```

Changing `RAM_BUILD_MANIFEST` changes the file inside the RAM++ wheel, so rebuild
and reinstall the wheel (README install steps) after a rename. The model
identities and vectors do not change.
