# Supplementary Appendix

This appendix has been provided by the authors to give readers additional information about their work.

Sheng Zhang, Yanbo Xu, Naoto Usuyama, et al. **A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs.** *NEJM AI.* DOI: 10.1056/AIoa2400640.

---

## Supplementary Note

### Details of BiomedCLIP architecture and ablation studies

#### Review of CLIP

We first give a brief review of the CLIP pretraining approach [7]. Given a batch of *N* (image, text) pairs, CLIP learns a multimodal embedding space by jointly training an image encoder and a text encoder to maximize the cosine similarity between the image and text embeddings of the *N* pairs in the batch while minimizing the cosine similarity of the embeddings of the other *N*² − *N* non-pairs. Concretely, CLIP minimizes the InfoNCE loss [23], i.e., a symmetric cross entropy loss over these similarity scores:

$$
\mathcal{L} = -\frac{1}{2N} \left( \sum_{i=1}^{N} \log \frac{e^{\cos(\boldsymbol{I}_i, \boldsymbol{T}_i)/\tau}}{\sum_{j=1}^{N} e^{\cos(\boldsymbol{I}_i, \boldsymbol{T}_j)/\tau}} + \sum_{i=1}^{N} \log \frac{e^{\cos(\boldsymbol{I}_i, \boldsymbol{T}_i)/\tau}}{\sum_{j=1}^{N} e^{\cos(\boldsymbol{I}_j, \boldsymbol{T}_i)/\tau}} \right)
$$

where τ is a learnable temperature parameter, directly optimized during training as a log-parameterized multiplicative scalar; **I**ᵢ and **T**ᵢ are embeddings for the *i*-th image and text, produced by a linear projection layer on top of the image encoder and text encoder. Rather than initializing with pretrained weights, CLIP trains the image encoder and text encoder from scratch. For the image encoder, CLIP considers two different architectures, ResNet50 [51] and Vision Transformer (ViT) [27]. The text encoder is effectively GPT-2 [24] based on transformer [52].

#### Adapting CLIP to BiomedCLIP

Biomedical text and images are drastically different from the web data used in CLIP pretraining. We find that the standard CLIP settings are suboptimal for biomedical vision-language pretraining. We thus conducted a systematic study of potential adaptations and identified a series of domain-specific adaptations for the biomedical domain. We used the optimization loss and cross-modal retrieval results on the validation set to guide our initial exploration and conducted detailed ablation studies.

On the text side, we replace a blank-slate GPT-2 with a pretrained language model more suited for biomedicine. Specifically, we initialize with PubMedBERT, which shows substantial gains from domain-specific pretraining [20]. Correspondingly, for the tokenizer, we replace Byte-Pair Encoding (BPE) [53] with WordPiece [54], which uses unigram-based likelihood rather than shattering all words to characters and greedily forming larger tokens based on frequency. The original CLIP uses a context of 77 tokens, but biomedical text is typically longer, as shown in Figure 1A. We thus increase the context size to 256, which covers 90% of PMC captions. Supplementary Table 1 shows that both modifications bring substantial improvements over the original CLIP model on the validation set.

**Supplementary Table 1:** Improvements from text-side domain-specific adaptations, as measured on the PMC-15M validation set. The training epochs are 8. All other hyperparameters are reported in Supplementary Table 9.

| text encoder | vocab | context length | loss (↓) | img2txt R@1 (↑) | txt2img R@1 (↑) |
|---|---|---|---|---|---|
| GPT | 50k general domain | 77 | 0.6626 | 64.53 | 63.56 |
| PubMedBERT | 30k domain specific | 77 | 0.5776 | 69.03 | 67.41 |
| PubMedBERT | 30k domain specific | 256 | 0.4807 | 73.50 | 72.26 |

On the image side, we first evaluated Vision Transformer (ViT) across different scales, ranging from ViT-Small, ViT-Medium, to ViT-Base. The suffix "/16" in the ViT model names refers to the patch size of 16×16 pixels, i.e., the input images are divided into patches of this size, and fed through the transformer blocks. As shown in Supplementary Table 2, we found that larger ViT results in better performance, confirming the importance of model scalability on our new dataset PMC-15M. We used the largest one (ViT-B/16) in all subsequent experiments.

**Supplementary Table 2:** Validation performance for various ViT models (Small, Medium, Base). All experiments use PubMedBERT to initialize the text encoder with the maximal context length of 256. The training epochs are 8. All other hyperparameters are reported in Supplementary Table 9.

| vision encoder | trainable params | hidden dim | loss (↓) | img2txt R@1 (↑) | txt2img R@1 (↑) |
|---|---|---|---|---|---|
| ViT-S/16 | 22M | 384 | 0.5342 | 69.45 | 68.02 |
| ViT-M/16 | 39M | 512 | 0.5063 | 71.85 | 70.22 |
| ViT-B/16 | 86M | 768 | 0.4807 | 73.50 | 72.26 |

**Supplementary Table 3:** Validation performance for vision encoders initialized with different weights. All experiments use PubMedBERT to initialize the text encoder, with the maximal context length of 256. All other hyperparameters are reported in Supplementary Table 9.

| vision encoder | initialization | loss (↓) | img2txt R@1 (↑) | txt2img R@1 (↑) |
|---|---|---|---|---|
| ViT-B/16 | random initialization | 0.3814 | 83.15 | 81.75 |
| ViT-B/16 | pretrained on ImageNet | 0.3819 | 82.90 | 81.86 |

Next, we compared two different ways to initialize the vision encoder. Supplementary Table 3 shows that the vision encoder pretrained on ImageNet [27] does not have advantages over random initialization. However, in our downstream tasks, ImageNet-pretrained weights offer more stable performance. Therefore, we chose to initialize ViT-B/16 with ImageNet-pretrained weights.

Lastly, biomedical image understanding often requires fine-grained visual features [30]. In Supplementary Table 4, we compared two choices of input image resolution: 224×224 and 384×384. By increasing image resolution, we observe significant gains in validation results. But this also leads to a doubling of pretraining time. In addition, increased image resolution does not consistently enhance performance in downstream tasks. As Supplementary Table 5 shows, BiomedCLIP exhibits inferior performance in zero-shot classification across all five datasets when using larger image size of 384 compared to 224. This discrepancy is particularly notable in PCam, where the raw image resolution (96×96) is considerably smaller than the model's input image size. Upsampling images here may introduce noise, potentially contributing to the observed decrease in performance. Consequently, we opt for an image size of 224 in the subsequent experiments.

**Supplementary Table 4:** Pretraining time and validation performance for vision encoders with different image sizes. All experiments use ViT-B/16 as the image encoder and PubMedBERT to initialize the text encoder, with a maximal context length of 256. All other hyperparameters are reported in Supplementary Table 9.

| image size | training time | loss (↓) | img2txt R@1 (↑) | txt2img R@1 (↑) |
|---|---|---|---|---|
| 224px | 1.00x | 0.3819 | 82.90 | 81.86 |
| 384px | 1.92x | 0.3406 | 84.63 | 83.56 |

**Supplementary Table 5:** Performance in downstream zero-shot image classification for vision encoders with different image sizes. AUROC (%) for TCGA-TIL; accuracy (%) for others. All hyperparameters are reported in Supplementary Table 9.

| image size | PCam | LC25000 (Lung) | LC25000 (Colon) | TCGA-TIL | RSNA | mean |
|---|---|---|---|---|---|---|
| 224px | 73.41 | 65.23 | 92.98 | 67.04 | 78.95 | 75.52 |
| 384px | 67.15 | 61.80 | 87.42 | 57.00 | 78.49 | 70.37 |

**Supplementary Table 6:** Validation performance with different batch size.

| batch size | img2txt R@1 (↑) | txt2img R@1 (↑) |
|---|---|---|
| 2k | 79.69 | 78.43 |
| 4k | 82.90 | 81.86 |

**Supplementary Table 7:** Validation performance with constant batch size of 4k for all 40 epochs vs increasing batch size from 4k (in the first 8 epochs) to 64k (in the remaining 32 epochs).

| batch size | img2txt R@1 (↑) | txt2img R@1 (↑) |
|---|---|---|
| 4k → 4k | 83.98 | 82.71 |
| 4k → 64k | 87.32 | 86.66 |

Finally, we investigated the impact of batch size. In Supplementary Table 6, we show larger batch size generally has better validation performance. In Supplementary Table 7, we studied increasing the batch size up to 64k to match the choices of [7, 55]. While there was a further increase in validation performance, we found that the gain did not translate to the downstream evaluation after reaching the batch size of 4k (Supplementary Table 6). The potential explanation for this could be that an extremely large batch size requires more training data and longer epochs. CLIP [7] uses 400M image-text pairs, while PMC-15M has 15M pairs. We chose the 4k batch size to train our BiomedCLIP.

#### Putting it all together

We pretrained a series of BiomedCLIP models on PMC-15M using the optimal batch schedule above and compared them with general-domain CLIP models [7]. As Supplementary Table 8 shows, large-scale pretraining or continual pretraining on PMC-15M is always helpful, and the best validation performance are generally attained using the biomedical pretrained language model (PubMedBERT), a larger vision transformer, and a higher image resolution. All hyperparameters are summarized in Supplementary Table 9.

**Supplementary Table 8:** Comparison of BiomedCLIP and general-domain CLIP models on validation performance. "WIT-400M → PMC-15M" indicates initialization with OpenAI CLIP weights that were pretrained on WIT-400M [7], followed by continual pretraining on PMC-15M. "PMB/256" denotes PubMedBERT with the maximal context length of 256.

| model | config | data | img2txt R@1 (↑) | txt2img R@1 (↑) |
|---|---|---|---|---|
| OpenAI CLIP | ResNet-50-224-GPT/77 | WIT-400M | 10.31 | 10.38 |
| OpenAI CLIP | ViT-B/16-224-GPT/77 | WIT-400M | 11.82 | 11.65 |
| BiomedCLIP | ResNet-50-224-GPT/77 | WIT-400M → PMC-15M | 81.17 | 80.17 |
| BiomedCLIP | ViT-B/16-224-GPT/77 | WIT-400M → PMC-15M | 81.57 | 80.89 |
| BiomedCLIP | ViT-B/16-224-BERT/256 | ImageNet/PubMed → PMC-15M | 82.90 | 81.86 |

**Supplementary Table 9:** Hyperparameters for pretraining settings.

| Hyperparameter | Value |
|---|---|
| optimizer | AdamW [56] |
| peak learning rate | 5.0e-4 |
| weight decay | 0.2 |
| optimizer momentum | β₁, β₂ = 0.9, 0.98 |
| eps | 1.0e-6 |
| learning rate schedule | cosine decay |
| epochs | 32 |
| warmup (in steps) | 2000 |
| random seed | 0 |
| image mean | (0.48145466, 0.4578275, 0.40821073) |
| image std | (0.26862954, 0.26130258, 0.27577711) |
| augmentation | RandomResizedCrop |
| validation frequency | every epoch |

#### Implementation

Our implementation is based on OpenCLIP [57], an open source software adapted for large-scale distributed training with contrastive image-text supervision. The pretraining experiments were conducted with up to 16 NVIDIA A100 GPUs or 16 NVIDIA V100 GPUs, via PyTorch DDP [58, 59]. To reduce memory consumption, we enable gradient checkpointing and automatic mixed precision (AMP) with datatype of bfloat16 (whenever supported by the hardware). In addition, we use a sharding contrastive loss [55], which achieves identical gradients to InfoNCE [23] and reduces memory usage by eliminating redundant computations and only computing the similarities of locally relevant features on each GPU.

---

**Supplementary Table 10:** Prompts used for zero-shot image classification.

| dataset | classes | templates |
|---|---|---|
| PCam | normal lymph node; lymph node metastasis | this is an image of {}; {} presented in image |
| LC25000 (Lung) | lung adenocarcinomas; normal lung tissue; lung squamous cell carcinomas | this is an image of {}; {} presented in image |
| LC25000 (Colon) | colon adenocarcinomas; normal colonic tissue | a photo of {}; {} presented in image |
| TCGA-TIL | none; tumor infiltrating lymphocytes | a photo of {}; {} presented in image |
| RSNA | normal lung; pneumonia | a photo of {}; {} presented in image |
