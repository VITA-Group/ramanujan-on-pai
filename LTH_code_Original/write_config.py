import json
import os
if __name__ == "__main__":
    config = {"data": "cifar10", "model": "resnet34"}
    paths = "results_lth"
    for density in os.listdir(paths):
        p = os.path.join(paths, density, 'cifar10')
        for model in os.listdir(p):
            pm = os.path.join(p, model, 'lth', '1')
            for ts in os.listdir(pm):
                with open(os.path.join(pm, ts, 'config.txt'), 'w') as handle:
                    config['model'] = model
                    json.dump(config, handle)
