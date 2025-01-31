## Project structure

fedsc_clef/
├── __init__.py
├── aggregator.py
├── client.py
├── data_utils.py
├── config.py
├── run_server.py
└── run_client.py

# How to Run

1. Start the Server (Aggregator):

```python -m fedscclef.run_server```

This will launch the aggregator on 0.0.0.0:8080 and wait for clients to connect.

2. Start Each Client in separate terminals or processes:

```python 
python -m fedsc_clef.run_client --cid 0
python -m fedsc_clef.run_client --cid 1
python -m fedsc_clef.run_client --cid 2
```

As each client connects, the aggregator will coordinate the rounds of training (“fitting”), do the Laplacian consensus, secure eigen decomposition, and eventually run evaluation at the final round.
