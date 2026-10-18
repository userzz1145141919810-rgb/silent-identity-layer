# A Silent Failure: When an Init Function Eats Your Fix

**A case study of a geometry layer that looked healthy for months while being mathematically inert.**

No GPU, no model, no dataset required to verify. `repro.py` proves the whole thing with the Python standard library alone.

---

## TL;DR

A residual "geometry layer" was inserted into a frozen 0.5B causal language model to test whether hyperbolic structure helps multi-hop reasoning. It ran across five datasets. It never crashed, never produced NaN, and its loss curve looked normal.

Run notes recorded two facts as if they were design properties:

> "gate stays near 0.5"
> "curvature stays near 0.7 - mild"

They were not design properties. A reset function, called at the start of every configuration, silently restored two parameters to `0.0`, undoing a fix applied in `__init__`.

With the boost parameter at zero, the layer collapsed to:

```
out = 0.5 * (0.5 * x / ||x||) + 0.5 * x
```

No curvature. No hyperbolic structure. And because `expmap` and `logmap` are exact inverses, the curvature cancelled out of the round-trip - its gradient was **identically zero**. It could never learn, and no amount of training would have changed that.

---

## 1. Symptoms - every signal was green

| Signal | Observation | Verdict at the time |
|---|---|---|
| Training loss | decreases normally | fine |
| NaN / Inf | none | fine |
| Gate value | ~0.5, stable across layers and steps | "as designed" |
| Curvature | ~0.7, stable | "mild, as designed" |
| vs. baseline | within noise on 5 datasets | "no harm" |
| vs. Euclidean control | nearly identical | "geometry adds little" |

Not one of these signals is false. That is the point. Each measurement was accurate; the interpretation was wrong.

---

## 2. The misread

Two observations got written into the experiment notes as architecture description:

**"gate stays near 0.5"** - actually `sigmoid(0.0)`. The gate logit was being reset to zero on every run.

**"curvature stays near 0.7"** - actually `softplus(0.01) + 1e-3 = 0.699`. The configured value was `0.01`, but it passed through `softplus` first. The number in the notes (0.7) and the number in the config (0.01) were never reconciled.

A hyperparameter that does not move for hundreds of steps is a suspect, not a design decision. This one never moved because it *could not* move.

---

## 3. Root cause - two sources of truth

The class defined the fix in its constructor:

```python
self.phi        = nn.Parameter(torch.tensor(0.8))  # fix: 0.0 made the boost identity
self.gate_logit = nn.Parameter(torch.tensor(1.0))  # fix: 0.0 diluted the signal
```

A separate reset function, called before every training configuration, overrode both:

```python
def reset_geo(cfg):
    for layer in geometry_layers:
        layer.raw_curvature.data.fill_(0.01)
        layer.phi.data.fill_(0.0)         # <-- overwrites the fix
        layer.gate_logit.data.fill_(0.0)  # <-- overwrites the fix
```

The constructor values were never used. Every run started from the broken state, then had only 150 optimizer steps to climb out - with a learning rate of 3e-3 and a global gradient clip shared with ~4M LoRA parameters.

The fix was correct. It was simply executed before the reset that undid it.

---

## 4. Numerical proof

`repro.py` reconstructs the layer with the Python standard library and measures it directly:

| Check | Result | Meaning |
|---|---|---|
| `boost(h, phi=0)` vs `h` | `0.000e+00` | strictly identity |
| `logmap(expmap(u))` vs `u` | `7.1e-13` | exact inverse - curvature cancels |
| Layer output vs `0.25*x/||x|| + 0.5*x` | `3.6e-13` | the layer is a fixed affine map |
| Curvature 0.01 -> 5.0, output change | `2.4e-13` | **zero gradient - curvature cannot learn** |
| Control: `phi=0.8`, curvature 0.01 -> 5.0 | `2.4e-01` | with a live boost, curvature matters |

The last row is the important one. The fix works. It was never in effect.

---

## 5. Why this stayed hidden for months

Three structural reasons, none of them carelessness:

1. **Nothing failed.** Silent degradation produces no error, no NaN, no divergence. There was nothing to catch.
2. **The symptom was stable.** `gate ~ 0.5` on every single run looked like a property of the architecture, not a symptom of a reset.
3. **The experiment still produced plausible numbers.** "No harm vs. baseline" looks publishable, and it was consistent across five datasets - because the layer really was near-identity on all five.

---

## 6. Five lessons

1. **A silent failure is more dangerous than a crash.** A crash costs an afternoon; a silent failure costs a quarter.
2. **Initialization and reset must share one source of truth.** If both `__init__` and `reset_*()` write the same parameter, they must read from the same constant - or the reset must be deleted.
3. **A hyperparameter that never moves is a suspect.** Print gradient norms, not just values.
4. **Do not let the diagnostic share assumptions with the thing it measures.** A "hyperbolicity" metric that normalises away the norm is measuring something else entirely.
5. **Capacity mismatch invalidates the experiment.** 72 learnable scalars cannot explain a 1% delta against ~4M LoRA parameters. The comparison was never capable of answering the question asked.

---

## 7. A 30-minute checklist

Run these before trusting any "no effect" result:

- [ ] Print `grad.norm()` for every learnable scalar every N steps. Exactly zero means dead, not small.
- [ ] `grep -n "\.data\.fill_(\|\.data *= *\|zero_()"` - diff every reset against the constructor defaults.
- [ ] Identity test: if `f(g(x)) == x`, the transform between them contributes nothing.
- [ ] Shuffle the inputs. A diagnostic that still reports the same number is measuring itself.
- [ ] Compare parameter counts. If A has 3 params and B has 4M, A cannot be responsible for a 1% delta.
- [ ] Assert that a parameter *should* move. If it has not moved in 100 steps, treat it as a bug until proven otherwise.

---

## 8. Reproduce

```bash
python3 repro.py
```

No dependencies. No GPU. Runs in under a second.

---

## License

MIT. Use it, teach with it, link it.

---

---

## 中文版（要点）

**一句话**：一个「双曲几何层」在五个数据集上跑出了「曲率随任务上升」的结论，但它的 boost 参数在每个配置开头被重置为 0，整层退化成固定仿射映射 `0.25*x/||x|| + 0.5*x`，曲率梯度在数学上恒等于 0。

**症状全绿**：loss 正常下降、无 NaN、gate 稳定 0.5、curvature 稳定 0.7、与 baseline 差异在噪声内。每一条测量都是真的，只有解释错了。

**根因**：`__init__` 里修好了（phi=0.8, gate=1.0），`reset_geo()` 里又打回 0.0。构造函数里的值一次都没生效。

**两条被误读的观察**：

- `gate ~ 0.5` 就是 `sigmoid(0.0)`
- `curvature ~ 0.7` 就是 `softplus(0.01) + 1e-3 = 0.699`（配置写的是 0.01，从没对上）

**五条教训**：

1. 静默失效比崩溃危险。崩溃费一个下午，静默失效费一个季度。
2. 初始化与重置必须单一真源。
3. 永远不动的超参数是嫌疑人，不是设计。
4. 诊断工具不能和被测对象共用假设。
5. 参数量差三个数量级以上时，别急着归因。

**检查清单**：见第 7 节，三十秒能跑完。

---

## 一句后记

这个 bug 不是「想错了」，是「改对了，然后被静默回滚了」。

在 AI 辅助写代码的时代，这类失效会越来越多：补丁写得越正确，被初始化函数覆盖时就越安静。
