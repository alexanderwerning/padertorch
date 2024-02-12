import click
from pathlib import Path
from padertorch import Configurable
import shutil
import paderbox as pb
import sys

BEATS_PATH = '/net/home/werning/projects/python_packages/unilm/beats'
assert Path(BEATS_PATH).exists(), f"BEATS_PATH {BEATS_PATH} does not exist"
sys.path.append(BEATS_PATH)

@click.command()
@click.argument("cls_name")
@click.argument("weight_file")
@click.argument("destination_path")
def create(cls_name, weight_file, destination_path):
    model_config = {"factory": cls_name, "pretrained_dir": weight_file, "load_config_from_checkpoint": True}
    model = Configurable.from_config(
        model_config,
    )
    destination_path = Path(destination_path)
    click.echo(f"Create model {cls_name} from {weight_file} at {destination_path}")
    assert not destination_path.exists()
    destination_path.mkdir(parents=True)
    checkpoint_dir = destination_path / "checkpoints"
    checkpoint_dir.mkdir()
    config_dir = destination_path / "1"
    # reset model to checkpoint and config
    # model_config.pop('pretrained_dir')
    # model_config.pop('load_config_from_checkpoint')
    model_config = model.get_config()
    config_content = {"factory": "padertorch.Trainer", "model": model_config}
    pb.io.dump(config_content, config_dir / "config.json")
    weight_file_name = Path(weight_file).name
    shutil.copy(weight_file, checkpoint_dir / weight_file_name)
    click.echo(f"Created model {weight_file_name} at {destination_path}")

if __name__ == '__main__':
    create()