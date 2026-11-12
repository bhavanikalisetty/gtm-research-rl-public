# GTM Research RL

This project tests how a policy qualifies fictional companies using a limited number of research actions. The policy has to cite evidence it actually saw, and the evaluator scores its decision separately.

The release includes 60 synthetic company scenarios across train, validation, and test splits, plus six duplicate-account challenges. The tools replay simulated observations. The data contains fictional companies and contacts, and the rules baseline runs without credentials or network access.

From the repository root, run with Python 3.10 or later:

```sh
python3 -m unittest research.environment.test_environment
python3 -m research.evaluation.run --split validation
```

The [environment guide](research/environment/README.md) explains the action and reward interfaces. The training directory contains the experimental SFT and RL runners. GPU training needs a compatible environment and a public model checkpoint. The hosted app, live integrations, provider captures, training logs, and deployment scripts are outside this release.
