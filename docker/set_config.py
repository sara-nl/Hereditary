"""Apply "key=value ..." overrides to [tool.flwr.app.config] in pyproject.toml.

The Flower run AND the scripts (centralised.py, compare.py, predict.py) read this table,
so overriding it here keeps all of them on the same settings. Usage:

    python set_config.py PYPROJECT "dataset='datasetA' event='NIV' min-coverage=0.4"

Values are TOML: strings in quotes ('datasetA'), numbers bare (0.4). Unknown keys and
invalid values are errors, so a typo cannot silently fall back to a default.
"""

import re
import shlex
import sys
import tomllib


def main(path, overrides):
    text = open(path).read()
    known = tomllib.loads(text)["tool"]["flwr"]["app"]["config"]
    start = text.index("[tool.flwr.app.config]")
    end = text.find("\n[", start + 1)
    end = len(text) if end < 0 else end
    section = text[start:end]

    for item in shlex.split(overrides, posix=False):
        if "=" not in item:
            sys.exit(f"RUN_CONFIG: '{item}' is not key=value")
        key, value = (s.strip() for s in item.split("=", 1))
        if key not in known:
            sys.exit(f"RUN_CONFIG: unknown key '{key}'. Known keys: {', '.join(known)}")
        try:
            parsed = tomllib.loads(f"x = {value}")["x"]
        except tomllib.TOMLDecodeError:
            sys.exit(f"RUN_CONFIG: {key}={value} is not a valid value (quote strings: {key}='...')")
        if isinstance(known[key], str) and not isinstance(parsed, str):
            sys.exit(f"RUN_CONFIG: {key} must be a quoted string, e.g. {key}='{value}'")
        if isinstance(known[key], float) and not isinstance(parsed, (int, float)):
            sys.exit(f"RUN_CONFIG: {key} must be a number, not {value}")
        if isinstance(known[key], float):
            value = repr(float(parsed))  # 1 -> 1.0, so Flower keeps the type
        section, n = re.subn(rf"(?m)^{re.escape(key)}\s*=.*$", lambda _: f"{key} = {value}", section)
        if n != 1:
            sys.exit(f"RUN_CONFIG: could not set {key} in {path}")
        print(f"config: {key} = {value}")

    open(path, "w").write(text[:start] + section + text[end:])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
