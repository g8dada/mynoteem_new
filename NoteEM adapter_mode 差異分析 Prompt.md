你現在要幫我對一個基於 **NoteEM / EM 架構**修改而來的程式碼 repository 做完整的工程分析。

## 背景

原始程式使用既有的 checkpoint，並透過 EM（Expectation-Maximization）相關流程執行。

我後來希望讓這套 EM framework 可以接上**另一種不同模型架構的 checkpoint**，因此在 config 中加入：

```python
adapter_mode: bool
```

其意義為：

- `adapter_mode = False`
  - 原始模式
  - 使用原本 NoteEM 預期的 checkpoint / model architecture
  - 應盡可能維持原始程式行為

- `adapter_mode = True`
  - Adapter 模式
  - 使用另一種不同 architecture 的 checkpoint
  - 為了讓新的 model/checkpoint 能夠接進原本 EM pipeline，我修改了 repository 中許多地方

目前的問題是：修改累積很多，我已經不完全記得每一項修改的目的、彼此關係，以及哪些修改是真的必要。

---

# 你的任務

請完整閱讀目前 repository 的程式碼，追蹤所有與：

```python
adapter_mode
```

直接或間接相關的邏輯，並建立一份**Artifact 格式的技術分析報告**。

這份報告不是單純列出 code diff。

我要你回答的核心問題是：

> 為了讓原本只支援舊 checkpoint 的 NoteEM / EM pipeline 能夠支援新的 checkpoint architecture，我到底改了哪些東西？每個修改解決了什麼 incompatibility？為什麼這樣改之後能 work？哪些地方其實可能仍然不 work？

---

# 分析原則

請把：

```python
adapter_mode=False
```

視為 **baseline / original pipeline**，

並把：

```python
adapter_mode=True
```

視為 **modified pipeline**。

從同一份程式碼中分別推導這兩條 execution path。

不要只搜尋：

```python
if adapter_mode:
```

還要繼續追蹤由它造成的 downstream differences，例如：

- 不同 model class
- 不同 checkpoint loading
- 不同 forward interface
- tensor shape transformation
- output key / dictionary 格式
- latent / logits / probability representation
- preprocessing
- postprocessing
- loss calculation
- E-step
- M-step
- pseudo-label / posterior calculation
- decoding
- batching
- masking
- device / dtype
- config
- model initialization
- inference mode
- parameter freezing
- normalization
- dimension mapping
- adapter / wrapper class
- utility function
- dataset / dataloader 行為

即使某段程式碼本身沒有出現 `adapter_mode`，只要它是為了讓 adapter pipeline 運作而修改，也要納入分析。

---

# 第一部分：Executive Summary

先用非常精簡的方式回答：

### 原本的問題是什麼？

說明：

> 新 checkpoint 與原本 NoteEM checkpoint 在哪些核心介面上不相容。

例如如果實際存在：

- architecture 不同
- state_dict 不同
- input representation 不同
- output representation 不同
- output dimensions 不同
- forward API 不同
- EM 所需要的 quantity 不同

請具體指出。

### Adapter Mode 的核心解法是什麼？

用 3–6 個重點概括整個 adaptation strategy。

不要先講 implementation detail。

我要先理解整個設計思想。

---

# 第二部分：True / False Pipeline 對照

建立一個清楚的比較表。

至少包含：

| Stage | adapter_mode=False | adapter_mode=True | 為什麼需要不同 |
|---|---|---|---|
| Config | | | |
| Model construction | | | |
| Checkpoint loading | | | |
| Input preprocessing | | | |
| Forward pass | | | |
| Output representation | | | |
| EM input | | | |
| E-step | | | |
| M-step | | | |
| Loss / objective | | | |
| Inference / decoding | | | |

如果某一項沒有差異，明確標示：

> No behavioral difference

不要為了填表而硬找差異。

---

# 第三部分：逐項列出所有修改

請按照「功能」分類，而不是按照檔案順序。

例如：

## 1. Model Loading Adaptation

### 原始行為

說明 `adapter_mode=False` 時：

- 使用什麼 model
- checkpoint 如何 load
- state_dict 如何 mapping
- model interface 是什麼

### Adapter 行為

說明 `adapter_mode=True` 時做了什麼不同處理。

### 為什麼需要這個修改？

請指出原本 incompatible 的地方。

不要只寫：

> 因為 checkpoint 不同。

而要寫清楚，例如：

> 原 checkpoint 的 forward() 回傳 shape 為 `[B, T, C]`，但新模型回傳的是 dictionary，其中 note logits 位於 `outputs["note"]`；原本 EM code 直接將 model output 視為 tensor，因此會在 XXX 發生錯誤。

### 如果不修改會怎樣？

具體描述：

- runtime error
- shape mismatch
- semantic mismatch
- silent bug
- training objective 錯誤
- EM 計算錯誤

### 修改後為什麼可以 work？

請沿著 tensor / variable 的 downstream usage 說明。

不要只因為「程式跑得動」就判定正確。

---

對每個 modification 都使用上述結構。

---

# 第四部分：Tensor / Data Flow

這一部分非常重要。

請分別畫出：

## `adapter_mode=False`

```text
raw input
→ preprocessing
→ model input
→ model
→ raw model output
→ transformation
→ EM representation
→ E-step
→ M-step
→ final output / loss
```

## `adapter_mode=True`

```text
raw input
→ preprocessing
→ adapter model input
→ new checkpoint/model
→ raw new-model output
→ adapter/transformation
→ EM-compatible representation
→ E-step
→ M-step
→ final output / loss
```

在每一步標示重要 tensor：

```text
name
shape
dtype（如果重要）
semantic meaning
```

例如：

```text
note_logits: [B, T, 88]
↓ sigmoid
note_prob: [B, T, 88]
↓ EM
posterior: [B, T, 88]
```

如果 shape 無法從 static code 確定，請寫：

> inferred / runtime-dependent

不要猜一個數字。

---

# 第五部分：找出 Adapter Layer 真正在做什麼

請回答：

> 新模型和 NoteEM 之間真正的「interface mismatch」到底是什麼？

並將 adapter 的功能拆成：

1. **Structural adaptation**
   - architecture / class / function API

2. **Representation adaptation**
   - logits / probability / feature / latent representation

3. **Dimensional adaptation**
   - tensor shape / dimensions / transpose / reshape

4. **Semantic adaptation**
   - 新模型某個 output 如何對應 NoteEM 所需要的 variable

5. **Optimization adaptation**
   - loss / gradient / frozen parameters / EM 更新

6. **Checkpoint adaptation**
   - loading / key mapping / strict=False / prefix removal 等

如果某一類不存在，直接說沒有。

---

# 第六部分：EM Compatibility Analysis

這部分不要只檢查 code 能不能執行。

請從數學與語意上檢查 adapter model 的 output 是否真的符合 EM 所需要的 quantity。

分析：

### E-step

- E-step 接收到什麼？
- 原模型提供的是什麼？
- adapter model 提供的是什麼？
- 兩者數學意義是否一致？
- 是否只是 shape 一樣，但 semantic 不同？

### M-step

- M-step optimize 哪些 parameters？
- adapter_mode=True 時 parameter set 是否改變？
- 是否有 freeze / detach？
- gradient 是否真的 flow 到預期的 model？
- checkpoint model 是否實際被更新？

### Objective

比較兩種 mode：

```text
adapter_mode=False objective
vs
adapter_mode=True objective
```

指出是否仍在 optimize 同一個 probabilistic / learning objective。

如果已經不是嚴格等價，請明確說：

> 這是一個 engineering approximation，而不是完全相同的 EM formulation。

---

# 第七部分：修改分類

請將所有修改分成四類：

### A. 必要修改

若不修改，adapter checkpoint 一定無法接入。

### B. Compatibility glue

主要用來做：

- reshape
- mapping
- wrapper
- API translation
- key conversion

### C. Defensive / engineering modifications

例如：

- device handling
- dtype
- exception handling
- logging
- optional config

它們不是 adapter 的核心，但提高 robustness。

### D. 可疑或可能不必要的修改

如果看到某段 code：

- 看不出必要性
- 與 adapter_mode 無真正關聯
- duplicate logic
- workaround 疑似已失效
- 很可能是 debug 過程留下來

請明確指出。

---

# 第八部分：Bug / Risk Audit

特別尋找以下問題：

### 1. False mode regression

我最在意的一點：

> 為了支援 `adapter_mode=True` 的修改，有沒有意外改壞 `adapter_mode=False`？

檢查原始 mode 是否仍維持原始 behavior。

### 2. Silent shape bugs

例如 broadcasting 沒有報錯，但計算其實錯誤。

### 3. Semantic mismatch

shape 正確但 variable 意義不同。

### 4. Gradient flow

檢查：

```python
detach()
no_grad()
requires_grad
eval()
train()
freeze
```

是否造成 adapter mode 無法真正執行 M-step。

### 5. Probability / logit misuse

例如：

```text
logits
probabilities
log probabilities
softmax
sigmoid
```

是否混用。

### 6. Mask / sequence length

兩種 architecture 對 temporal resolution 的定義是否一致。

### 7. Checkpoint loading

特別檢查：

```python
strict=False
```

或 key filtering 是否可能讓重要 weight 根本沒有被 load。

### 8. Train / inference mismatch

adapter model 的 preprocessing 或 forward 是否與它原本 training checkpoint 的設定一致。

---

# 第九部分：Work / Not Work 判定

最後不要只給「看起來可以」。

對每個主要 adaptation 給出判定：

### ✅ Correct

可以從 code logic 明確驗證。

### ⚠️ Likely works, but assumption exists

程式邏輯合理，但依賴某個假設。

請說明假設。

### ❌ Incorrect / inconsistent

可以明確找到問題。

### ❓ Cannot verify statically

需要 runtime information。

例如：

```text
實際 tensor shape
checkpoint metadata
sample model output
training configuration
```

請列出要如何驗證。

---

# 第十部分：最終總結

最後提供：

## Adapter Mode 修改地圖

用簡短 dependency chain 表示，例如：

```text
New checkpoint
↓
different model architecture
↓
different forward output
↓
output adapter
↓
NoteEM-compatible representation
↓
existing E-step
↓
modified M-step
```

接著回答這五個問題：

1. `adapter_mode=True` 最核心的三個修改是什麼？
2. 哪些修改只是為了 compatibility，而沒有改變演算法？
3. 哪些修改實際改變了原本 EM 的數學或 optimization behavior？
4. 目前最可能出 bug 的三個地方在哪？
5. `adapter_mode=False` 是否仍能被認為是原始 implementation？

---

# Artifact 的寫作要求

請將完整分析整理成一份 **Artifact 技術文件**。

風格要求：

- 技術精確
- 高資訊密度
- 不冗言贅字
- 不要寫大量背景科普
- 不要把 code 原封不動貼一大段
- 使用表格、流程圖、code reference 幫助理解
- 每一項結論盡量附上實際程式位置

引用程式碼時使用：

```text
file_path.py :: ClassName.function_name()
```

必要時加上相關 code snippet，但只保留關鍵幾行。

---

# 非常重要：不要只根據變數名稱推測

每當你說：

> 這個修改是為了 XXX

都必須追蹤 downstream code，確認它實際如何被使用。

區分：

```text
Fact:
可以直接由程式碼證明

Inference:
根據 execution flow 推導，很可能是如此

Uncertain:
只看 static code 無法確認
```

如果無法確定，不要自行腦補。

---

# 分析順序

請按照以下順序工作：

1. 找出 `adapter_mode` 的 config 定義與所有 reference
2. 建立 `False` execution path
3. 建立 `True` execution path
4. 找出兩者 first divergence point
5. 從 divergence point 一路追蹤到 pipeline 最後
6. 找出為了兼容新 checkpoint 而產生的 helper / wrapper / transformation
7. 比較 tensor semantics
8. 分析 E-step / M-step
9. 檢查 gradient flow
10. 檢查 False mode regression
11. 建立完整修改清單
12. 產生 Artifact 報告

不要一開始就直接寫報告。

先完整理解 execution flow，再整理結論。