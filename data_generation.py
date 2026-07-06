import mcgpu_pet_wrapper as mpw
import pet_sim_gen as psg
from pet_sim_gen.examples import sf_proxy      # NOTE: flat module, not a submodule

cfg = mpw.default_config()
bounds = psg.suggest_bounds(cfg)

psg.generate_dataset(
    n=1, config=cfg, bounds=bounds, out_dir="../MCGPU_data",
    stratify_key=sf_proxy,             # any recipe -> float works
    stratify_target=(0.05, 0.3, 5),  # flatten over the key in [0.05, 0.3], 5 bins
)