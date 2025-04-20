import json
from glob import glob
import os.path as osp
import yaml

def get_metric_from_json(json_file, term_key="AUC"):
    content = json.load(open(json_file, "r"))
    # import pdb; pdb.set_trace()
    
    metric_list = []
    term_out_list = []
    for node in content['nodes']:
        # import pdb; pdb.set_trace()
        if isinstance(node, dict) and isinstance(node.get('metric', dict()), dict):
            metric_list.append(node.get('metric', dict()).get("value", None))
        if isinstance(node, dict) and "_term_out" in node:
            term_out = node.get("_term_out", "")
            term_out = [t for t in term_out if term_key in t]
            term_out_list.append(term_out)    
    return metric_list, term_out_list


def get_goal_from_config(file_path):
    """
    Reads a YAML file and extracts the 'goal' field.

    :param file_path: Path to the YAML file.
    :return: The value of the 'goal' field, or None if it doesn't exist.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.unsafe_load(file)
            return data.get('goal', 'None')
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return 'None'
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the YAML file. {e}")
        return 'None'



if __name__ == "__main__":
    log_dir_list = sorted(glob(osp.join(osp.dirname(__file__), "logs", "*-*_*")), key=lambda x: int(osp.basename(x).split("-")[0]))
    for log_dir in log_dir_list:
        # print(log_dir)
        log_file = osp.join(log_dir, "journal.json")
        metric, term_out = get_metric_from_json(log_file)

        goal = get_goal_from_config(osp.join(log_dir, "config.yaml"))
        if ("odir" in log_dir):
            goal = "ODIR-5K"
        else:
            goal = "MedNIST"

        # print key metrics
        log_dir = osp.basename(log_dir)
        # print(log_dir, goal, " ".join([str(m) for m in metric]))
        # print(log_dir.replace("2-base-", "").replace("_2d", ""), goal)
        print(f"[{log_dir}]", goal, max([0.0] + [float(m) for m in metric if m is not None]))
        # metric, term_out = get_metric_from_json(log_file, term_key="Acc")
        # print(f"[{log_dir}]", term_out[:])
        # metric, term_out = get_metric_from_json(log_file, term_key="AUC")
        # print(f"[{log_dir}]", term_out[:])
        print("----")
