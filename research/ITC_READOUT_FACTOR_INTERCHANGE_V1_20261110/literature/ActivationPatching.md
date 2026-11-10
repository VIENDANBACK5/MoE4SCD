# Towards Best Practices of Activation Patching in Language Models: Metrics and Methods

Fred Zhang and Neel Nanda, ICLR2024. [Accepted paper page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html), [primary PDF](https://openreview.net/pdf?id=9eJv5PS27Q).

The paper studies how intervention details and evaluation metrics affect activation-patching findings. Its experiments concern language models, not crowns. It supports methodological caution about interpreting a patching result, not an empirical claim about this vision carrier. Our factor swap is an established intervention-style diagnostic and is not represented as a faithful reproduction of this paper or a novel intervention principle.

We fix both swaps, all output metrics and the decision bands before new forwards; preserve native-output equivalence checks; and explicitly analyze factorization/gauge ambiguity. Final-mask recovery after a patch is distinguished from an inference-valid single-carrier improvement.
