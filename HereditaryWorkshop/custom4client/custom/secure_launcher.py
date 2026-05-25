import logging
from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError

class SecureJobLauncher(ClientProcessJobLauncher):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def launch_job(self, job_meta: dict, fl_ctx: FLContext):
        # Check authorization result set by security handler
        auth_result = fl_ctx.get_prop(FLContextKey.AUTHORIZATION_RESULT)
        if auth_result is False:
            reason = fl_ctx.get_prop(FLContextKey.AUTHORIZATION_REASON, "Security Check Failed")
            self.logger.error("SecureJobLauncher: Job launch aborted due to security violation: %s", reason)
            raise UnsafeComponentError(f"Job BLOCKED: {reason}")
        
        self.logger.info("SecureJobLauncher: Job authorized, launching...")
        return super().launch_job(job_meta, fl_ctx)
