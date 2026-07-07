import os
import sys
from config import get_config
from voxcpm import VoxCPM


def main() -> None:
    config = get_config()
    print(f"Downloading VoxCPM2 from {config['model_id']} to {config['cache_dir']}")
    model = VoxCPM.from_pretrained(
        config["model_id"],
        load_denoiser=False,
        cache_dir=config["cache_dir"],
        local_files_only=False,
        optimize=False,
        device="cpu",
    )
    print("Model cached successfully")


if __name__ == "__main__":
    main()
