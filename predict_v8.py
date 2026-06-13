"""V8: fine-step denoising (10 internal Langevin steps vs 4 in V3)."""
import os, sys, time
import numpy as np
import jittor as jt
from omegaconf import OmegaConf

jt.flags.use_cuda = 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.vm import VelocityModule, patch_based_denoise


def main():
    vm_ckpt = "experiments/vm/checkpoint_199.pkl"
    test_dir = "dataset_test_noisy/shapenet"
    output_dir = "tmp_predict/v8"

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
    print(f"Test samples: {len(test_samples)}")
    N_total = len(test_samples)

    t0 = time.time()
    for idx, sample in enumerate(test_samples):
        noisy_path = os.path.join(test_dir, sample, "noisy.npy")
        pc_noisy = np.load(noisy_path).astype(np.float32)

        pc_tensor = jt.array(pc_noisy)
        with jt.no_grad():
            pc_denoised = patch_based_denoise(
                model=vm, pcl_noisy=pc_tensor,
                patch_size=1000, seed_k=6, seed_k_alpha=1,
            )
        result = pc_denoised.detach().numpy()

        out_dir = os.path.join(output_dir, "dataset_test_noisy", "shapenet", sample)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "predict.npy"), result.astype(np.float32))

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (N_total - idx - 1)
            print(f"  {idx+1}/{N_total}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Done! {N_total} samples in {elapsed:.0f}s")
    print("Fine-step: 10 internal Langevin steps per patch_based_denoise call")


if __name__ == "__main__":
    main()
