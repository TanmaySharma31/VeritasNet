# Multimodal Feature Reference

## Text Features (TF-IDF)
* **Dimensions:** 5,000
* **Why TF-IDF:** It acts as a lightweight, memory-safe text representation compared to heavy transformers like BERT, allowing this pipeline to train on limited Colab GPUs.
* **Bigrams Used:** By using `ngram_range=(1, 2)`, the model captures short phrases (like "under fire") which helps mitigate literal vs. figurative mismatches.

## Visual Features (ResNet-18)
* **Dimensions:** 512
* **Precomputed Caching:** Passing images through the CNN takes heavy GPU VRAM. By precomputing the 512-D vectors under `torch.no_grad()` and caching them as `.npy` files, our final Multimodal MLP trains in seconds rather than hours.
