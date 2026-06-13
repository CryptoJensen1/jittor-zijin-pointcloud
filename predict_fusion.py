"""Multi-scale point cloud denoising: fuses 1000-patch (good CD) + 2000-patch (good P2S) results."""
import os, sys, time, argparse
import numpy as np
from scipy.spatial import cKDTree
import jittor as jt
from omegaconf import OmegaConf

jt.flags.use_cuda = 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.vm import VelocityModule, patch_based_denoise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='tmp_predict/fusion')
    args = parser.parse_args()

    vm_ckpt = "experiments/vm/checkpoint_199.pkl"
    test_dir = "dataset_test_noisy/shapenet"
    patch_sizes = [1000, 1500, 2000]

    # Load configs and model
    vm_cfg = OmegaConf.to_container(OmegaConf.load("configs/model/vm.yaml"))
    transform_cfg = OmegaConf.to_container(OmegaConf.load("configs/transform/vm.yaml"))

    print(f"Loading VM: {vm_ckpt}")
    vm = VelocityModule(model_config=vm_cfg, transform_config=transform_cfg)
    vm.load(vm_ckpt)
    vm.eval()
    vm.set_predict(True)

    # Scan test samples
    test_samples = []
    for root, dirs, files in os.walk(test_dir):
        if "noisy.npy" in files:
            test_samples.append(os.path.relpath(root, test_dir))
    test_samples.sort()
    print(f"Test samples: {len(test_samples)}")
    N_total = len(test_samples)

    t0 = time.time()
    for idx, sample in enumerate(test_samples):
        noisy_path = os.path.join(test_dir, sample, "noisy.npy")
        pc_noisy = np.load(noisy_path).astype(np.float32)
        N_pts = pc_noisy.shape[0]

        results = []
        for ps in patch_sizes:
            pc_tensor = jt.array(pc_noisy)
            with jt.no_grad():
                pc_next = pc_tensor
                for _ in range(args.num_steps):
                    pc_next = patch_based_denoise(
                        model=vm, pcl_noisy=pc_next,
                        patch_size=ps, seed_k=6, seed_k_alpha=1,
                    )
                results.append(pc_next.detach().numpy())

        # Three-scale KNN fusion: anchor on 1000-patch (best CD), match
        # 1500 and 2000 patch results, then average all three.
        tree1 = cKDTree(results[1])
        tree2 = cKDTree(results[2])
        _, nn1 = tree1.query(results[0], k=1)
        _, nn2 = tree2.query(results[0], k=1)
        fused = (results[0] + results[1][nn1] + results[2][nn2]) / 3.0

        # Save
        out_dir = os.path.join(args.output_dir, "dataset_test_noisy", "shapenet", sample)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "predict.npy"), fused.astype(np.float32))

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (N_total - idx - 1)
            print(f"  {idx+1}/{N_total}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Done! {N_total} samples in {elapsed:.0f}s")
    print(f"Fusion: patch_sizes={patch_sizes}")


if __name__ == "__main__":
    main()
