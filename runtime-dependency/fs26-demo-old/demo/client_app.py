"""demo client app."""

from flwr.app import Context
from flwr.clientapp import ClientApp
from flwr.common import Message, RecordDict, ConfigRecord
import xgboost as xgb
import logging
# Flower ClientApp
app = ClientApp()


@app.query()
def query(msg: Message, context: Context):
    """Query the model on local data."""
    logging.basicConfig(filename="../MY_CLIENTLOG.txt", level=logging.INFO)
    logging.info("Hello from Flower!!")
    logging.info("My xgboost version is: ", xgb.__version__)


    # Return the numpy version in the message content
    import nvflare

    res = RecordDict()
    res.config_records["info"] = ConfigRecord({"xgboost_version": xgb.__version__, "nvflare_version": nvflare.__version__})
    return Message(content=res, reply_to=msg)

