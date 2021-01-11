import sys
import json
from pathlib import Path

default_config_file = Path(sys.argv[1])
other_config_file = Path(sys.argv[2])

default_config = json.loads(default_config_file.read_text())
other_config = json.loads(other_config_file.read_text())


def remove_equal_values(default_dict, other_dict):
    for key in other_dict:
        if key in default_dict and default_config[key] == other_config[key]:
            other_dict.pop(key)
        else:
            if key in default_dict and isinstance(default_dict[key], dict) and isinstance(other_dict[key], dict):
                remove_equal_values(default_dict[key], other_dict[key])


remove_equal_values(default_config, other_config)

print(other_config)