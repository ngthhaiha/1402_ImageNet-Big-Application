# Graph 3DResNet2

This folder contains a graph-enabled variant of `3dresnet2.py`.

Pipeline:

```text
motion / optical flow crop sequence
-> 3DResNet motion_encoder
-> bbox graph interaction module
-> graph-aware motion feature
-> memory
-> motion reconstruction score

appearance crop + reconstructed motion
-> frame prediction branch
-> frame prediction score

memory attention
-> relation / self-labeling score

final anomaly score = motion score + frame score + relation score
```

Entrypoint:

```bash
python graph_3dresnet2/graph_3dresnet2.py --mode train --dataset_name ped2
python graph_3dresnet2/graph_3dresnet2.py --mode test --dataset_name ped2 --checkpoint <checkpoint>
```

Important options:

```text
--graph_layers
--graph_topk
--graph_alpha
--graph_proximity_weight
--relation_score_weight
--disable_graph
--disable_frame_grouping
```

`--disable_graph` keeps the original 3DResNet2 behavior for ablation. `--disable_frame_grouping` disables graph-friendly batches; leave it off for graph training/testing.
