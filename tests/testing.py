import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)

import tests.example_networks as en
from framework.attack import Injector
from framework.criterion import classification_accuracy
import pandas as pd

os.chdir(current_dir)

nets = ["nn", "cnn", "rnn"]


for net in nets:
    print(f"\n#####{net}#####\n")
    # Get a trained simple NN model directly
    model, X_train, X_test, y_train, y_test, train_fn, eval_fn = en.get_example_network(net, train=True, epochs=500)

    # Evaluate right away
    acc = eval_fn(model, X_test, y_test)
    print(f"Trained {net} accuracy: {acc:.4f}")

    inj = Injector(model, X = X_test, y = y_test, criterion=classification_accuracy)

    results = inj.run_seu(0)

    results = pd.DataFrame(results)
    print("Average criterion post seu:", results["criterion_score"].mean())