# VeritasNet - Viva Quick Reference

## 1. Project Objective
A multimodal machine learning system that analyzes whether a news image contextually matches its headline.

## 2. Architecture Breakdown (Method B)
* **Text Branch:** TF-IDF Vectorizer (5,000 max features, 1-2 n-grams) -> Linear(5000, 256) -> ReLU -> Dropout.
* **Image Branch:** ResNet-18 CNN (Final classification layer removed, 512-D output) -> Linear(512, 256) -> ReLU -> Dropout.
* **Fusion Strategy:** Early-fusion via Concatenation. Text (256) + Image (256) = 512-D shared latent space.
* **Classifier MLP:** Linear(512, 64) -> ReLU -> Dropout -> Linear(64, 1 logit output).

## 3. Key Examiner Questions & Answers
**Q: Why didn't you train ResNet-18 from scratch?**
A: Training an 11-million parameter CNN from scratch on 7,000 images would lead to extreme overfitting. Using ImageNet pretrained weights gave us robust visual feature extraction instantly.

**Q: Why did you only fine-tune `layer4` of ResNet-18?**
A: To prevent 'catastrophic forgetting.' The early layers already know how to detect edges and shapes. Fine-tuning only the final residual block allows the network to adapt to journalistic photos without destroying its foundational visual knowledge.

**Q: How did you prevent data leakage?**
A: We strictly deduplicated the dataset before splitting. More importantly, the TF-IDF vectorizer and ResNet image feature standardizations were fitted **only on the training split**. The validation and test sets were strictly transformed, never fitted on.
