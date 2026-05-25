# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom Site Security Handler for NVIDIA FLARE
==============================================

This component plugs into the NVFlare event system and intercepts two key
moments in the job lifecycle:

  • EventType.SUBMIT_JOB  - fired on the FL server when a job is first
    submitted by an admin/user.
  • EventType.DEPLOY_JOB_TO_CLIENT - fired on each client site just before
    the job app is deployed to that node.

At each of these hooks the handler:
  1. Resolves the job's app root directory from the FLContext.
  2. Walks every Python file inside that directory.
  3. Applies regex-based pattern matching (data_leak_patterns) for
     network/file exfiltration, sensitive-credential access, and
     encoding/obfuscation primitives.
  4. Optionally runs an AST walk for dangerous imports or built-ins
     (exec / eval / __import__).
  5. Sets FLContextKey.AUTHORIZATION_RESULT (bool) and
     FLContextKey.AUTHORIZATION_REASON (str) so the NVFlare engine
     knows whether to proceed.

Drop this file into  <workspace>/local/custom/security_handler.py
and register it in    <workspace>/local/resources.json
(see resources.json in the same directory for the config snippet).
"""

import ast
import logging
import os
import re

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext

logger = logging.getLogger(__name__)


class CustomSecurityHandler(FLComponent):
    """Site-local security handler that scans job code for data-leak patterns.

    Registered as a component in local/resources.json so the NVFlare runtime
    instantiates it at startup on every participating site (server and clients).

    The handler listens for:
      - EventType.SUBMIT_JOB         (server-side, on submission)
      - EventType.DEPLOY_JOB_TO_CLIENT (client-side, before deployment)

    If a violation is detected the job is blocked:
      - FLContextKey.AUTHORIZATION_RESULT is set to False
      - FLContextKey.AUTHORIZATION_REASON carries a human-readable message

    If the code is clean:
      - FLContextKey.AUTHORIZATION_RESULT is set to True
    """

    # ---------------------------------------------------------------------------
    # Pattern catalogue
    # ---------------------------------------------------------------------------
    DATA_LEAK_PATTERNS: dict[str, list[str]] = {
        "network_exfiltration": [
            r"requests\.(post|get|put|patch)",
            r"urllib\.(request|parse)",
            r"http\.client",
            r"socket\.",
            r"ftplib\.",
            r"smtplib\.",
        ],
        "file_exfiltration": [
            r"shutil\.copy",
            r'open\(.*["\']w["\']',        # open(path, 'w') / open(path, "w")
            r"pickle\.dump",
            r"json\.dump.*external",
            r"csv\.writer",
        ],
        "sensitive_data_access": [
            r"os\.environ",
            r"getpass\.",
            r"secrets\.",
            r"\bconfig\.",
            r"\.password\b",
            r"\.token\b",
            r"\.key\b",
        ],
        "encoding_obfuscation": [
            r"base64\.",
            r"hashlib\.",
            r"cryptography\.",
            r"\bencrypt\b",
            r"\bdecrypt\b",
        ],
    }

    # Imports that are outright banned (checked in the AST pass)
    BANNED_IMPORTS: set[str] = {"socket", "requests", "urllib", "ftplib", "smtplib"}

    # Built-in calls considered dangerous
    DANGEROUS_BUILTINS: set[str] = {"exec", "eval", "__import__"}

    # ---------------------------------------------------------------------------
    # FLComponent interface
    # ---------------------------------------------------------------------------

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        """Route NVFlare lifecycle events to the appropriate check."""
        if not "heartbeat" in event_type:
            self.logger.info("CustomSecurityHandler: received event '%s'", event_type)

        if event_type == EventType.SUBMIT_JOB:
            # Server-side: job just received from admin
            self.logger.info("CustomSecurityHandler: intercepting SUBMIT_JOB event")
            self._run_security_check(fl_ctx, phase="submit_job")

        elif event_type in [EventType.DEPLOY_JOB_TO_CLIENT, EventType.BEFORE_JOB_LAUNCH, "AFTER_JOB_DEPLOY"]:
            # Client-side: job app is about to be/has been deployed to this node
            self.logger.info(
                "CustomSecurityHandler: intercepting job deployment/launch event: %s", event_type
            )
            self._run_security_check(fl_ctx, phase=event_type)

    # ---------------------------------------------------------------------------
    # Core security orchestration
    # ---------------------------------------------------------------------------

    def _run_security_check(self, fl_ctx: FLContext, phase: str) -> None:
        """Perform the full security check and write the result into fl_ctx."""

        # Retrieve the Job ID
        # The official method is get_job_id(), but during early event phases
        # like _before_job_launch, NVFlare stores it under "__run_num__".
        job_id = None
        if hasattr(fl_ctx, "get_job_id"):
            job_id = fl_ctx.get_job_id()
            
        if not job_id:
            job_id = fl_ctx.get_prop("__run_num__")

        # Retrieve the app root from the context
        app_root: str | None = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        if not job_id:
            try:
                # Use INFO level so we can actually see this in the logs!
                keys = fl_ctx.get_prop_keys()
                self.logger.info("CustomSecurityHandler: Could not find job_id. Available keys: %s", keys)
            except Exception:
                self.logger.info("CustomSecurityHandler: Could not find job_id and failed to list keys.")

        if app_root is None or "startup/.." in app_root:
            workspace_root: str | None = fl_ctx.get_prop(FLContextKey.WORKSPACE_ROOT)
            if workspace_root and job_id:
                potential_job_root = os.path.join(workspace_root, str(job_id))
                if not os.path.exists(potential_job_root):
                    potential_job_root = os.path.join(workspace_root, f"run_{job_id}")
                
                app_root = potential_job_root

        if not app_root or not os.path.exists(app_root):
            self.logger.warning(
                "CustomSecurityHandler [%s]: Could not determine valid app_root (job_id=%s); skipping code scan.",
                phase, job_id
            )
            self._set_authorized(fl_ctx, reason="app_root not found")
            return

        self.logger.info(
            "CustomSecurityHandler [%s]: scanning code at '%s'", phase, app_root
        )

        ok = self.validate_code_content(app_root)

        if ok:
            self._set_authorized(fl_ctx, reason="All code checks passed")
            self.logger.info(
                "CustomSecurityHandler [%s]: AUTHORIZED – no violations found", phase
            )
        else:
            reason = (
                "Job BLOCKED: suspicious code patterns detected during security scan. "
                "Check logs for details."
            )
            self._set_denied(fl_ctx, reason=reason)
            self.logger.warning(
                "CustomSecurityHandler [%s]: DENIED – %s", phase, reason
            )

    # ---------------------------------------------------------------------------
    # Public helper – validate_code_content
    # ---------------------------------------------------------------------------

    def validate_code_content(self, code_path: str) -> bool:
        """Validate Python code for security and data-leak patterns.

        Args:
            code_path: Path to a directory containing Python source files.

        Returns:
            True  – code is clean (job may proceed).
            False – a violation was detected (job should be blocked).
        """
        if not os.path.exists(code_path):
            return True  # No custom code is OK

        for root, _dirs, files in os.walk(code_path):
            for filename in files:
                if not filename.endswith(".py"):
                    continue

                file_path = os.path.join(root, filename)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                        content = fh.read()
                except OSError as exc:
                    self.logger.error(
                        "CustomSecurityHandler: cannot read '%s': %s", file_path, exc
                    )
                    return False  # Unreadable file → treat as violation

                # --- Regex scan --------------------------------------------------
                if self.detect_data_leaks(content, self.DATA_LEAK_PATTERNS):
                    return False

                # --- AST scan ----------------------------------------------------
                try:
                    tree = ast.parse(content, filename=file_path)
                except SyntaxError as exc:
                    self.logger.error(
                        "CustomSecurityHandler: syntax error in '%s': %s",
                        file_path,
                        exc,
                    )
                    return False  # Invalid Python syntax → block

                if self.check_ast_for_leaks(tree):
                    return False

        return True  # All files passed

    # ---------------------------------------------------------------------------
    # Regex pattern detection
    # ---------------------------------------------------------------------------

    def detect_data_leaks(self, content: str, patterns: dict[str, list[str]]) -> bool:
        """Detect potential data-leak patterns in source code text.

        Args:
            content:  Full text of a Python source file.
            patterns: Dict mapping category name → list of regex patterns.

        Returns:
            True if any pattern matches (violation detected).
        """
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, content, re.IGNORECASE):
                    self.logger.warning(
                        "CustomSecurityHandler: potential data-leak detected "
                        "[category=%s] [pattern=%s]",
                        category,
                        pattern,
                    )
                    return True
        return False

    # ---------------------------------------------------------------------------
    # AST-based leak detection
    # ---------------------------------------------------------------------------

    def check_ast_for_leaks(self, tree: ast.AST) -> bool:
        """Walk the AST looking for dangerous imports or built-in calls.

        Args:
            tree: A compiled AST (from ast.parse).

        Returns:
            True if a suspicious pattern is found (violation detected).
        """
        for node in ast.walk(tree):
            # ---- Dangerous top-level imports ------------------------------------
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    if top_level in self.BANNED_IMPORTS:
                        self.logger.warning(
                            "CustomSecurityHandler: banned import detected: '%s'",
                            alias.name,
                        )
                        return True

            # ---- 'from X import ...' where X is banned --------------------------
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in self.BANNED_IMPORTS:
                    self.logger.warning(
                        "CustomSecurityHandler: banned 'from' import: 'from %s ...'",
                        node.module,
                    )
                    return True

            # ---- Dangerous built-in function calls ------------------------------
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.DANGEROUS_BUILTINS:
                        self.logger.warning(
                            "CustomSecurityHandler: dangerous built-in call "
                            "detected: '%s()'",
                            node.func.id,
                        )
                        return True

        return False

    # ---------------------------------------------------------------------------
    # FLContext helpers
    # ---------------------------------------------------------------------------

    def _set_authorized(self, fl_ctx: FLContext, reason: str = "") -> None:
        """Record an AUTHORIZED result in the FLContext."""
        fl_ctx.set_prop(FLContextKey.AUTHORIZATION_RESULT, True, private=False, sticky=True)
        fl_ctx.set_prop(FLContextKey.AUTHORIZATION_REASON, reason, private=False, sticky=True)

    def _set_denied(self, fl_ctx: FLContext, reason: str) -> None:
        """Record a DENIED result in the FLContext."""
        fl_ctx.set_prop(FLContextKey.AUTHORIZATION_RESULT, False, private=False, sticky=True)
        fl_ctx.set_prop(FLContextKey.AUTHORIZATION_REASON, reason, private=False, sticky=True)
