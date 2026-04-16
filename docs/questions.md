# Question For Group

I noticed that the stage-1 attention KL loss for the reinspection module was very large during InternVL training.

Example logs:

```text
Epoch 1 Step 350: loss=100.1671 ce=0.5344 attn=199.2655 lr=5.71e-06
Epoch 1 Step 400: loss=112.8803 ce=0.4108 attn=224.9390 lr=6.64e-06
Epoch 1 Step 450: loss=103.2002 ce=0.5166 attn=205.3673 lr=7.44e-06
Epoch 1 Step 500: loss=127.5863 ce=0.4445 attn=254.2834 lr=8.23e-06
Epoch 1 Step 550: loss=96.6825 ce=0.4152 attn=192.5346 lr=9.03e-06
Epoch 1 Step 600: loss=108.7608 ce=0.4901 attn=216.5413 lr=9.96e-06
```

Right now the total loss is:

```text
loss = ce_loss + stage1_attn_loss_weight * attn_loss
```

and for InternVL stage 1:

```text
stage1_attn_loss_weight = 0.5
```

So the attention term dominates the total loss.

The issue seems to be that the KL loss was computed after expanding the target to all queries, and then using `batchmean`, which divides only by batch size. Since we have `n_queries = 64`, the logged KL is effectively summed over all queries instead of averaged per query.

I changed it to normalize per query:

```python
loss = F.kl_div(pred.log(), target, reduction="batchmean")
return loss / pred.shape[1]
```

Questions:

1. Does averaging the stage-1 KL over queries look correct to you for the reinspection module?
2. Would you keep KL for InternVL stage 1, or switch to something else like focal or MSE?
3. After per-query normalization, should we keep `stage1_attn_loss_weight = 0.5`, or retune it?
4. Conceptually, do we want every query to match the same bbox target distribution, or should queries be diversified the way we do for Qwen focal supervision?
