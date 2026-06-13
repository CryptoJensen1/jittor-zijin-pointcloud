"""Predict with DistanceModule: loads VM + DM checkpoints and denoises test set."""
import os, sys, time, argparse
import numpy as np
import jittor as jt
from omegaconf import OmegaConf

jt.flags.use_cuda = 1

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.vm import VelocityModule, patch_based_denoise
from src.model.distance import DistanceModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=1000)
    args = parser.parse_args()

    vm_ckpt = "experiments/vm/checkpoint_199.pkl"
    dm_ckpt = "experiments/distance/checkpoint_49.pkl"
    test_dir = "dataset_test_noisy/shapenet"
    output_dir = "tmp_predict/dm"
    num_steps = args.num_steps
    patch_size = args.patch_size

    # Load configs
    vm_cfg = OmegaConf.to_container(OmegaConf.load("configs/model/vm.yaml"))
    dm_cfg = OmegaConf.to_container(OmegaConf.load("configs/model/distance.yaml"))
    transform_cfg = OmegaConf.to_container(OmegaConf.load("configs/transform/vm.yaml"))

    # Create VM, load checkpoint
    print(f"Loading VM: {vm_ckpt}")
    vm = VelocityModule(model_config=vm_cfg, transform_config=transform_cfg)
    vm.load(vm_ckpt)
    vm.eval()
    vm.set_predict(True)

    # Create DM, load checkpoint
    print(f"Loading DM: {dm_ckpt}")
    dm = DistanceModule(model_config=dm_cfg, transform_config=transform_cfg)
    dm.load(dm_ckpt)
    dm.eval()
    dm.set_predict(True)

    # Attach DM to VM
    vm.set_distance_module(dm)
    print(f"DistanceModule attached. num_steps={num_steps}, patch_size={patch_size}")

    # Scan test data
    test_samples = []
    for root, dirs, files in os.walk(test_dir):
        if "noisy.npy" in files:
            rel_dir = os.path.relpath(root, test_dir)
            test_samples.append(rel_dir)
    test_samples.sort()
    print(f"Test samples: {len(test_samples)}")

    # Denoise each sample
    t0 = time.time()
    for idx, sample in enumerate(test_samples):
        noisy_path = os.path.join(test_dir, sample, "noisy.npy")
        pc_noisy = np.load(noisy_path).astype(np.float32)
        N = pc_noisy.shape[0]

        # Convert to Jittor tensor
        pc_tensor = jt.array(pc_noisy)  # (N, 3)

        with jt.no_grad():
            pc_next = pc_tensor
            for _ in range(num_steps):
                pc_next = patch_based_denoise(
                    model=vm,
                    pcl_noisy=pc_next,
                    patch_size=patch_size,
                    seed_k=6,
                    seed_k_alpha=1,
                )
            pc_denoised = pc_next.detach().numpy()

        # Ensure shape matches
        assert pc_denoised.shape[0] == N, f"Shape mismatch: {pc_denoised.shape[0]} != {N}"

        # Save
        out_dir = os.path.join(output_dir, "dataset_test_noisy", "shapenet", sample)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "predict.npy"), pc_denoised.astype(np.float32))

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(test_samples) - idx - 1)
            print(f"  {idx+1}/{len(test_samples)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Done! {len(test_samples)} samples in {elapsed:.0f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
