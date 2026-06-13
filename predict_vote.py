"""Test-time voting inference for PCT ModelNet40 classification."""
import os, json, argparse
import numpy as np
import jittor as jt
from pct import PCT

jt.flags.use_cuda = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='pct_model.pkl')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--n_points', type=int, default=2048)
    parser.add_argument('--num_votes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', type=str, default='result.json')
    args = parser.parse_args()

    # Load trained model
    model = PCT(num_classes=40)
    model.load(args.ckpt)
    model.eval()
    print(f"Model loaded from {args.ckpt}")

    # Load test data from disk
    test_data = np.load(os.path.join(args.data_dir, 'test_points.npy'))
    N = len(test_data)
    n_cached = test_data.shape[1]
    print(f"Test samples: {N}, Points per sample: {n_cached}")
    print(f"Voting: {args.num_votes} votes per sample")

    results = {}
    bs = args.batch_size

    with jt.no_grad():
        for start in range(0, N, bs):
            end = min(start + bs, N)
            batch_pts = test_data[start:end].copy()  # (B, 2048, 3)
            B = end - start

            # Sample n_points
            replace = n_cached < args.n_points
            choice = np.random.choice(n_cached, args.n_points, replace=replace)
            batch_pts = batch_pts[:, choice, :]

            # Collect logits from multiple votes
            logits_sum = None
            for v in range(args.num_votes):
                aug_pts = batch_pts.copy()

                # Y-axis random rotation
                angle = np.random.uniform(0, 2 * np.pi)
                cy, sy = np.cos(angle), np.sin(angle)
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
                aug_pts = aug_pts @ Ry.T

                # Mild jitter for diversity
                jitter = np.random.randn(*aug_pts.shape).astype(np.float32) * 0.005
                aug_pts = aug_pts + jitter

                aug_pts = aug_pts.transpose(0, 2, 1)  # (B, 3, N)
                aug_pts = jt.array(aug_pts)

                logits = model(aug_pts)
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits

            avg_logits = logits_sum / args.num_votes
            preds = avg_logits.argmax(dim=1)[0]

            for i in range(B):
                sample_id = start + i
                results[str(sample_id)] = int(preds[i].item())

            if (start // bs + 1) % 20 == 0:
                print(f"  {end}/{N} samples done")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {args.output}")
    print("Done!")


if __name__ == '__main__':
    main()
