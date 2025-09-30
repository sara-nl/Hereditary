---
tags: [advanced, differential_privacy, privacy]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Differential privacy

The following steps describe how to use Flower's built-in Secure Aggregation components. This example demonstrates how to apply `SecAgg+` to the same federated learning workload as in the [quickstart-pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example. The `ServerApp` uses the [`SecAggPlusWorkflow`](https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html#secaggplusworkflow) while `ClientApp` uses the [`secaggplus_mod`](https://flower.ai/docs/framework/ref-api/flwr.client.mod.secaggplus_mod.html#flwr.client.mod.secaggplus_mod). To introduce the various steps involved in `SecAgg+`, this example introduces as a sub-class of `SecAggPlusWorkflow` 

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!NOTE\]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.25"
```


```bash
flwr run . --run-config is-demo=false
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
