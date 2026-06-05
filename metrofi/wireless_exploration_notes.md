# MetroFi wireless exploration 中文说明

这个文件解释 `data/wireless_exploration.ipynb` 里展示的 MetroFi dataset generation 流程，并说明当前训练、sampling、evaluation 分别使用 processed dataset 里的哪些内容。

## 1. 原始数据是什么

notebook 默认读取：

```text
data/metrofi/stumble_filtered.txt
```

每一行被读成四列：

```text
mac lat lon rssi
```

含义是：

- `mac`: 被扫描到的 AP/MAC 地址，也就是后面 graph 的 node identity。
- `lat`, `lon`: 扫描发生的位置；一个唯一 `(lat, lon)` 后面会变成一张 graph。
- `rssi`: 这个 AP 在这个位置的信号强度读数。

notebook 里显示：

```text
Loaded 200694 samples.
Unique MAC addresses: 70
Unique locations: 30991
```

也就是说，原始数据是 200,694 条无线扫描记录，不是直接的 graph dataset。代码会把这些 scan rows 聚合成 graph。

## 2. `Unique MAC addresses: 70` 是什么

`Unique MAC addresses: 70` 表示整个 scan log 里一共出现过 70 个不同 AP/MAC 地址。

在当前 MetroFi 建模里，这 70 个 MAC 地址构成固定的全局 node universe：

```text
node 0  -> 第 0 个 MAC 地址
node 1  -> 第 1 个 MAC 地址
...
node 69 -> 第 69 个 MAC 地址
```

正式脚本会把每个 MAC 地址映射成整数 `mac_id`，并保存到：

```python
metadata["mac_id_to_address"]
```

当前 `data/metrofi/metrofi.pkl` 里，前几个映射是：

```text
0 -> 00:0A:DB:01:74:E0
1 -> 00:0A:DB:01:86:00
2 -> 00:0A:DB:01:8F:60
```

训练时，每张 graph 都会变成固定大小的 adjacency matrix：

```text
70 x 70
```

其中 `(i, j)` 永远表示第 `i` 个 AP 和第 `j` 个 AP 之间的 interference。这个固定坐标系很重要，因为它让不同 location 的 graph 可以对齐到同一组 AP 上。

## 3. `Unique locations: 30991` 是什么

`Unique locations: 30991` 表示原始 scan log 里有 30,991 个不同经纬度点：

```python
(lat, lon) -> location_id
```

MetroFi 的 graph 定义是：

```text
一个 location = 一张 graph
```

所以 30,991 个 unique locations 会对应 30,991 张 per-location graphs。每张 graph 描述：

```text
在这个地理位置上，哪些 AP 被观测到了，以及这些 AP 两两之间的 inferred interference。
```

当前 processed pickle 保存了全部 30,991 张 location graphs，原始 split 是：

```text
train: 24793 graphs
val:    3099 graphs
test:   3099 graphs
```

但训练 loader 还会过滤掉 observed AP 数少于 3 的 graph。当前配置：

```python
min_observed_nodes = 3
```

过滤后实际用于 train/val/test 的数量是：

```text
train: 7623 graphs
val:    945 graphs
test:   943 graphs
```

被过滤掉的 graph 通常只观测到 0、1、2 个 AP，边太少，不适合训练 weighted graph generation。

## 4. `metrofi.pkl` 里面有什么

正式数据生成脚本输出：

```text
data/metrofi/metrofi.pkl
```

顶层结构是：

```python
{
    "train": [...],
    "val": [...],
    "test": [...],
    "metadata": {...},
}
```

其中：

- `train`, `val`, `test`: NetworkX graph list。
- 每个 graph 对应一个 location。
- `metadata`: 全局 MAC 地址表、location 坐标、interference 单位等信息。

当前 metadata 包括：

```python
metadata = {
    "mac_id_to_address": list of 70 MAC addresses,
    "location_id_to_coords": dict of 30991 location coordinates,
    "max_interference": -52.0,
    "max_interference_watts": 6.30957344480193e-09,
    "interference_units": "dbm",
}
```

每张 graph 的 graph-level attributes 类似：

```python
graph.graph = {
    "location_id": 12726,
    "coordinates": (45.521511, -122.675987),
}
```

每个 node 是整数 `mac_id`，node attributes 类似：

```python
graph.nodes[i] = {
    "mac_address": "...",
    "observed": True or False,
    "mean_rssi": ...,
    "mean_rssi_raw": ...,
    "mean_rssi_dbm": ...,
    "sample_count": ...,
}
```

这里最重要的是 `observed`：

- `observed=True`: 这个 AP 在当前 location 真的被扫描到了。
- `observed=False`: 这个 AP 只是 70-node universe 里的占位节点，在当前 location 没有被观测到。

每条 edge 的 attributes 类似：

```python
graph.edges[u, v] = {
    "weight": raw_interference,
    "interference": raw_interference,
    "interference_watts": ...,
    "interference_dbm": ...,
    "mean_rssi_a_dbm": ...,
    "mean_rssi_b_dbm": ...,
}
```

刚生成 pickle 时，`weight` 是 raw interference。当前数据单位是 dBm，所以值大概在：

```text
[-95.0, -52.0]
```

训练 loader 读入后会把 `weight` 改成 normalized `[0, 1]`，同时把原始值保存成 `interference_raw`。

## 5. 每个 location 如何变成 graph

notebook 的核心函数是：

```python
build_location_graphs(...)
```

它按 `location_id` groupby。对每个 location：

1. 找出这个 location 里观测到的 AP。
2. 对每个 AP 计算 mean RSSI。
3. 把 RSSI 转成 dBm，再转成 Watts。
4. 对 observed AP 两两组合，建立 edge。
5. edge weight 是两个 AP 的 inferred interference。

notebook 版本里，每张 graph 只包含当前 location 实际观测到的 AP。

正式脚本里，每张 graph 会先加入所有 70 个 AP：

```python
for mac_id in all_mac_ids:
    graph.add_node(mac_id, observed=False, ...)
```

然后把当前 location 真正观测到的 AP 标成 `observed=True`。这就是为什么训练 batch 里需要 `observed_mask`。

## 6. RSSI 到 interference 的转换

notebook 使用：

```python
rssi_raw_to_dbm(rssi_raw) = rssi_raw - 95.0
```

再把 dBm 转成 Watts：

```python
dbm_to_w(dbm) = 0.001 * 10 ** (dbm / 10.0)
```

edge 构造时，对同一个 location 里的所有 observed AP 两两组合：

```python
for mac_a, mac_b in combinations(observed_ids, 2):
```

支持三种 interference 模型：

```python
min:     min(p_a, p_b)
sum:     p_a + p_b
product: p_a * p_b
```

当前默认使用：

```python
model = "min"
output = "dbm"
```

所以 edge 上保存的 `interference` 是 dBm 单位，可能是负数。notebook 统计到：

```text
Global min interference: -95.0
Global max interference: -52.0
```

这不是错误，dBm 本来就常见负值。

## 7. 训练时用到什么

训练入口：

```bash
python main.py train --model metrofi --device cuda:0
```

训练通过 `WirelessDatasetModule` 读取：

```text
data/metrofi/metrofi.pkl
```

训练真正用到：

- graph 的 edge `weight`
- node 的 `observed`
- metadata 里的 MAC 数量，也就是 70

加载时会做：

1. 读取 `train/val/test` graph list。
2. 根据所有 split 的 raw edge value 计算全局 min/max。
3. 把 edge `weight` normalize 到 `[0, 1]`。
4. 把原始物理值保存成 `interference_raw`。
5. 根据 node 的 `observed` 生成 `observed_mask`。
6. 把 graph list 转成 tensor dataset。

训练 batch 是：

```python
(X, adj, observed_mask)
```

当前配置中 batch size 是 128，所以实际 shape 是：

```text
X:             (128, 70, 1)
adj:           (128, 70, 70)
observed_mask: (128, 70)
```

含义：

- `X`: node feature，现在基本是初始化的 ones feature。
- `adj`: normalized weighted adjacency，边权在 `[0, 1]`。
- `observed_mask`: 每张 graph 哪些 AP 在该 location 被观测到了。

MetroFi 配置里：

```python
training_mode = "weighted"
```

所以训练 loss 是 weighted MSE。模型不是预测 binary edge/no-edge，而是在 observed AP 子图上预测 normalized interference weight。

## 8. sampling / generation 时用到什么

generation 入口：

```bash
python main.py gen --model metrofi --device cuda:0
```

主要用到：

- `metadata["mac_id_to_address"]`: 确定 fixed AP universe 大小，当前是 70。
- datamodule 的 `interference_range`: 把生成的 `[0, 1]` 权重 rescale 回原始 dBm range。
- test graphs: 用于 evaluation metrics。

sampling 不是条件生成某一个具体 location。它生成完整的 70-node weighted adjacency：

```text
generated adj: 70 x 70, values in [0, 1]
```

然后 `run_gen_wireless` 会把 normalized value 转回原始 interference range：

```python
adj_rescaled = adj_samples * interference_scale + interference_min
```

当前数据中：

```text
interference_min = -95.0
interference_max = -52.0
```

所以生成结果会从 `[0, 1]` 被映射回大约 `[-95, -52]` dBm。

## 9. evaluation 时用到什么

`WirelessSamplingMetrics` 比较 generated graphs 和 test graphs。

它主要用到：

- generated graph 的 `interference_raw` 或 `weight`
- reference/test graph 的 `interference_raw` 或 `weight`
- graph 的 node/edge structure

主要指标：

- `edge_ks`: 对同一对 AP 的 interference 分布做 KS 距离。
- `edge_wasserstein`: 对同一对 AP 的 interference 分布做 Wasserstein 距离。
- `degree_weighted`: weighted degree histogram MMD。
- `spectre`: spectral MMD。

这里的“同一对 AP”依赖固定的 70-node universe。例如 `(11, 33)` 这对 AP 会在很多 location 中出现，evaluation 会比较这对 AP 在真实数据和生成数据中的 interference 分布。

因为 generated graph 是完整 70-node graph，而真实 test graph 每个 location 只 observed 一部分 AP，所以 metrics 里还会按照 test graph size distribution 从 generated graph 中 sample subgraphs，用于 structural comparison。

## 10. notebook 里的可视化

notebook 后半部分主要是 sanity check：

- graph size histogram: 看每个 location 观测到多少 AP。
- location graph plot: 看单个 location 的 AP 连接结构。
- adjacency matrix plot: 把 graph 放到固定 `70 x 70` AP 坐标系里。
- edge-wise interference histogram: 看同一对 AP 在不同 location 下的 interference 分布。
- isolated AP check: 检查是否有 AP 从未参与任何 edge。

notebook 最后显示：

```text
Num isolated APs: 0
Isolated APs: []
```

说明过滤后的数据里，所有出现过的 AP 都至少与其他 AP 有过共现边。

## 11. 总结

MetroFi dataset setup 可以概括为：

```text
raw rows: mac, lat, lon, rssi
  -> factorize MAC and location
  -> one location becomes one graph
  -> node = AP/MAC
  -> edge = two APs co-observed at that location
  -> edge weight = inferred interference from RSSI
  -> fixed 70-node universe
  -> observed_mask marks which APs were observed in each location
  -> train on normalized edge weights in [0, 1]
```

所以，`70` 是全局 AP/MAC 数量；`30991` 是 location graph 的总数量；训练看到的是 fixed-size weighted adjacency plus observed mask；sampling 生成的是完整 70-node weighted graph，再 rescale 回物理 interference 范围做 evaluation。

## 12. 关于 location-conditional generation 和 CFG 的分析

现在这套 MetroFi 训练更接近 unconditional generation：模型学习的是整体的 `p(weighted AP graph)`，也就是从所有真实 locations 的 graph distribution 里学一个总体分布。虽然每张真实 graph 都有 `location_id` 和 `coordinates`，而且 node identity 在所有 graph 之间是对齐的，但当前训练并没有把 location 信息作为 condition 输入模型。因此 sampling 出来的 graph 没有“这是某个具体 location”的语义，它只是一个看起来像真实 MetroFi location graph 的样本。

从 wireless application 的角度看，location-conditional generation 是更自然的目标。数据本身其实暗含的是 `p(weighted AP graph | location)`：给定一个经纬度位置，哪些 AP 会被观测到、这些 AP 之间的 interference 强度是多少，都应该和空间位置有关。所以如果目标是做更有物理意义的生成或预测，那么仅仅做 unconditional generation 可能不够。更合理的方向是让模型看到 location condition，例如 normalized `(lat, lon)` 或者由坐标编码出来的 embedding。

不过，不建议直接用 `location_id` embedding 作为 condition。`location_id` 有 30,991 个，而且很多 location 只出现一次，用离散 ID 很容易让模型记忆训练集，而不是学习可泛化的空间规律。更稳妥的 condition 是连续坐标 `(lat, lon)`，或者基于坐标构造的 spatial features。第一版可以先把 normalized coordinates 经过一个小 MLP 得到 condition embedding，然后拼到 GraphTransformer 的 global feature `y` 里。当前模型本来就有 global feature `y`，diffusion time `t` 也是拼到 `y` 上的，所以这是和现有框架最兼容的改法。

从 Jacobi diffusion 的数学角度看，conditional generation 并不需要改变 Jacobi SDE 本身。Jacobi SDE 负责的是在 `[0, 1]` edge weights 上定义 forward diffusion 和 reverse diffusion。做 conditional generation 时，我们真正要改的是 score network：从 unconditional score `score(A_t, t)` 变成 conditional score `score(A_t, t, c)`。也就是说，reverse Jacobi SDE 的形式可以保持不变，只是 drift correction 里用的 score 变成带 condition 的 score。

Classifier-free guidance 也可以放进这个框架里。CFG 的核心形式是把 conditional score 和 unconditional score 做线性组合：

```text
score_guided(A_t, t, c)
= score_uncond(A_t, t)
  + guidance_scale * (score_cond(A_t, t, c) - score_uncond(A_t, t))
```

这个公式并不是 DDPM 独有的；只要 reverse process 需要一个 score，就可以考虑用这种方式增强 conditional direction。在当前 Jacobi reverse SDE 里，代码最终会用 score 修正 drift，大致形式是 `drift = drift - diffusion^2 * score`。因此最数学一致的 CFG 插入点就是 score 本身：先算 conditional score，再算 unconditional score，然后组合成 guided score，最后交给 Jacobi reverse SDE。

需要注意的是，当前 MetroFi weighted 模型不是 direct score model。它训练时更像 denoising / reconstruction：模型预测 clean edge weight 或 clean `A_0`，然后 `JacobiScore` 再根据预测的 `A_0` 和 Jacobi transition density 计算 score。因此 CFG 有两个可能插入点。一个简单做法是在 `A_0` prediction 上做 guidance，即分别预测 `A0_cond` 和 `A0_uncond`，然后组合出 `A0_guided`。这个实现容易，但数学上更像 heuristic，因为 Jacobi score 对 `A_0` 的依赖是非线性的，`score(A_t, guided_A0, t)` 不严格等于 guided score。

更合理的做法是在 score level 做 CFG。也就是分别用 conditional model output 计算 `score_cond`，用 null-condition model output 计算 `score_uncond`，然后在线性组合 score：

```text
score_guided = score_uncond + s * (score_cond - score_uncond)
```

这样更贴近 CFG 的原始定义，也更符合 Jacobi reverse SDE 需要 score 的结构。代价是 sampling 每一步需要跑两次模型，一次带 condition，一次用 null condition，因此会更慢。训练时则需要做 condition dropout：以一定概率，比如 10% 到 20%，把真实 condition 替换成 null condition，让同一个模型同时学 conditional 和 unconditional 两种行为。

对 MetroFi 来说，null condition 可以是全零 coordinate embedding，也可以是 learned null embedding。condition 本身第一版建议用 normalized `(lat, lon)`，不要用 `location_id`。如果后面发现单纯坐标信号太弱，可以再考虑更强的 spatial condition，例如附近 training locations 的 AP observation frequency、附近 AP-pair interference prior，或者 kNN spatial summary。但这些高级 spatial priors 要注意不能泄漏 test split 信息，最好只从 train split 构造。

还有一个独立但很关键的问题是 `observed_mask`。MetroFi 的真实生成过程其实可以拆成两部分：先决定这个 location 会观测到哪些 AP，也就是 `p(mask | location)`；再决定这些 observed AP 之间的 weights，也就是 `p(weights | mask, location)`。当前模型训练时使用真实 `observed_mask` 做 masking，但 sampling 时更像是在生成完整 70-node weighted graph。因此，即使加入 location condition，模型可能仍然没有真正学会“这个 location 应该有哪些 AP 被观测到”。更完整的 conditional generation 可能需要先建一个 mask model，从 location 预测或采样 70 维 observed mask，然后 Jacobi diffusion 只在这个 mask 对应的子图上生成 weights。

所以比较稳的路线是分阶段做。第一阶段，只做 coordinate-conditioned weighted generation，把 coordinates 加到 global feature `y`，并实现 score-level CFG，验证 conditional signal 是否能改善 edge-wise interference distribution 或 location-wise reconstruction metrics。第二阶段，再单独建模 `p(observed_mask | location)`，让 sampling 不再总是 full 70-node graph，而是先生成或预测 location-specific observed AP set。这样比一开始同时解决 mask 和 weight 两个问题更稳。

总结来说，CFG 在 Jacobi diffusion 里是数学上合理的，前提是最好在 score level 做，而不是只在 `A_0` prediction level 做。Jacobi SDE 本身不用改；需要改的是 score network 的 condition 输入、训练时的 condition dropout、sampling 时的 conditional/unconditional 双分支 score 计算。对于 MetroFi，location-conditional generation 是合理方向，但不要直接用 `location_id`，优先用 coordinates 或 carefully designed spatial features，并且后续需要认真处理 observed mask 的条件生成问题。
