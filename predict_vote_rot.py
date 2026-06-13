"""Rotation voting: denoise from 5 angles, rotate back, KNN-average."""
import os, sys, time
import numpy as np
from scipy.spatial import cKDTree
import jittor as jt
from omegaconf import OmegaConf

jt.flags.use_cuda = 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.vm import VelocityModule, patch_based_denoise


def rot_matrix_y(deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def main():
    vm_ckpt = "experiments/vm/checkpoint_199.pkl"
    test_dir = "dataset_test_noisy/shapenet"
    output_dir = "tmp_predict/vote"
    angles = [0, 72, 144, 216, 288]

    vm_cfg = OmegaConf.to_container(OmegaConf.load("configs/model/vm.yaml"))
    transform_cfg = OmegaConf.to_container(OmegaConf.load("configs/transform/vm.yaml"))

    print(f"Loading VM: {vm_ckpt}")
    vm = VelocityModule(model_config=vm_cfg, transform_config=transform_cfg)
    vm.load(vm_ckpt)
    vm.eval()
    vm.set_predict(True)

    test_samples = []
    for root, dirs, files in os.walk(test_dir):
        if "noisy.npy" in files:
            test_samples.append(os.path.relpath(root, test_dir))
    test_samples.sort()
    print(f"Test samples: {len(test_samples)}, angles: {angles}")
    N_total = len(test_samples)

    t0 = time.time()
    for idx, sample in enumerate(test_samples):
        noisy_path = os.path.join(test_dir, sample, "noisy.npy")
        pc_noisy = np.load(noisy_path).astype(np.float32)

        results = []
        for angle in angles:
            R = rot_matrix_y(angle)
            pc_rot = pc_noisy @ R.T
            pc_tensor = jt.array(pc_rot)
            with jt.no_grad():
                pc_denoised_rot = patch_based_denoise(
                    model=vm, pcl_noisy=pc_tensor,
                    patch_size=1000, seed_k=6, seed_k_alpha=1,
                )
            pc_back = pc_denoised_rot.detach().numpy() @ R
            results.append(pc_back)

        # KNN-match all to result[0], then average
        fused = results[0].copy()
        for j in range(1, len(results)):
            tree = cKDTree(results[0])
            _, nn = tree.query(results[j], k=1)
            fused += results[j][nn]
        fused /= len(results)

        out_dir = os.path.join(output_dir, "dataset_test_noisy", "shapenet", sample)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "predict.npy"), fused.astype(np.float32))

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (N_total - idx - 1)
            print(f"  {idx+1}/{N_total}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Done! {N_total} samples in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
