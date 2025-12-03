---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:24678
- loss:MultipleNegativesRankingLoss
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Does long distance relationship work?
  sentences:
  - How can I maintain my long distance relationship to the best of my ability?
  - Consider the automobile manufacturing industry. Why is it that the production
    of vehicles is always active? Will there ever come a day when the demand comes
    down to nil?
  - How many water bottles can you carry in one trip (two arms)?
- source_sentence: Which best and cheap badminton racket should I buy?
  sentences:
  - How does 3rd wave feminism affect men’s rights?
  - What are the best ways to lose weight, especially around your core?
  - Which is the best Li Ning badminton racket I can buy?
- source_sentence: How do you become a web/developer?
  sentences:
  - What do I need to do to become a web developer?
  - How many times a day should you meditate?
  - How does your height increase?
- source_sentence: What if the South won the American Civil War?
  sentences:
  - Which is better, League Of Legends or Dota 2?
  - What would have happened if the American civil was lost?
  - Is the new Macbook Pro 2016 an over priced disappointment?
- source_sentence: What is your favorite anime and why?
  sentences:
  - Why do I feel like I'm going to die soon?
  - What's your favorite anime? And why?
  - What if the Dirty War in Argentina never happpened?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Quora Eval
      type: Quora-Eval
    metrics:
    - type: cosine_accuracy@1
      value: 0.0
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.8733509234828496
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.941952506596306
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.978891820580475
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.0
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.2911169744942832
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.1883905013192612
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.09788918205804747
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.0
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.8733509234828496
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.941952506596306
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.978891820580475
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.5847482126290905
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.4488508816015412
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.4503179191964502
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What is your favorite anime and why?',
    "What's your favorite anime? And why?",
    'What if the Dirty War in Argentina never happpened?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9863, -0.0040],
#         [ 0.9863,  1.0000, -0.0156],
#         [-0.0040, -0.0156,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `Quora-Eval`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.0        |
| cosine_accuracy@3   | 0.8734     |
| cosine_accuracy@5   | 0.942      |
| cosine_accuracy@10  | 0.9789     |
| cosine_precision@1  | 0.0        |
| cosine_precision@3  | 0.2911     |
| cosine_precision@5  | 0.1884     |
| cosine_precision@10 | 0.0979     |
| cosine_recall@1     | 0.0        |
| cosine_recall@3     | 0.8734     |
| cosine_recall@5     | 0.942      |
| cosine_recall@10    | 0.9789     |
| **cosine_ndcg@10**  | **0.5847** |
| cosine_mrr@10       | 0.4489     |
| cosine_map@100      | 0.4503     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Datasets

#### Unnamed Dataset

* Size: 6,678 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 6 tokens</li><li>mean: 14.28 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 14.06 tokens</li><li>max: 43 tokens</li></ul> |
* Samples:
  | sentence_0                                                                               | sentence_1                                                                 |
  |:-----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------|
  | <code>What do you think about Donald Trump and his campaign?</code>                      | <code>What do you think about Donald Trump in September?</code>            |
  | <code>In what way do you think the reservation system can be diminished in India?</code> | <code>How can this reservation system can be overthrown from India?</code> |
  | <code>"What inspired ""Angels and Demons"" by Dan Brown?"</code>                         | <code>What inspired Dan Brown to write Angels and Demons?</code>           |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

#### Unnamed Dataset

* Size: 18,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 6 tokens</li><li>mean: 15.87 tokens</li><li>max: 71 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 15.87 tokens</li><li>max: 64 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.39</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                         | sentence_1                                                                              | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:-----------------|
  | <code>I am about to start my gate CSE 2018 preparation from November, what are the some tips that I should keep in my mind?</code> | <code>How do I start preparation for CSE 2018 as I am currently in Second Year? </code> | <code>0.0</code> |
  | <code>Is there an ultimate limit to how much information the human brain can actually hold?</code>                                 | <code>Is there a limit to how much knowledge a human brain can retain?</code>           | <code>1.0</code> |
  | <code>How did smartphones changed the life of elderly?</code>                                                                      | <code>How has a smartphone changed your life?</code>                                    | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | Quora-Eval_cosine_ndcg@10 |
|:-----:|:----:|:-------------------------:|
| -1    | -1   | 0.5915                    |
| 1.0   | 418  | 0.5847                    |


### Framework Versions
- Python: 3.13.0
- Sentence Transformers: 5.1.2
- Transformers: 4.57.3
- PyTorch: 2.9.1+cpu
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->