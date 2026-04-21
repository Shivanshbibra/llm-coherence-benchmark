import os, re, csv, time, math, itertools, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
 
warnings.filterwarnings("ignore")
 

 
OPENROUTER_API_KEY = "API_KEY"   
 
# Models: display_name → OpenRouter model ID
MODELS = {
    "GPT-4o":   "openai/gpt-4o",
    "Gemini":   "google/gemini-2.0-flash-001",
    "DeepSeek": "deepseek/deepseek-chat",
}
 
PROMPT_CSV = "all_1000_prompts.csv"   # the single file you upload to Colab
OUTPUT_DIR = "results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
HOGWARTS = {"gryffindor", "hufflepuff", "ravenclaw", "slytherin"}
 
MODEL_COLORS = {
    "GPT-4o":   "#2563EB",
    "Gemini":   "#16A34A",
    "DeepSeek": "#DC2626",
}
 

 
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
 
def query_model(model_id: str, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 * (attempt + 1)
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                print(f"    [FAIL] {model_id}: {e}")
                return ""
    return ""
 
# ─────────────────────────────────────────────────────────────
# CELL 5 — LOAD PROMPTS
# The CSV has columns: id, category, prompt, difficulty,
#   constraint_count, steps, paraphrase_group, source
# The script reads ALL metadata directly from the CSV —
# you do NOT need to change anything here.
# ─────────────────────────────────────────────────────────────
 
def load_prompts() -> pd.DataFrame:
    if not os.path.exists(PROMPT_CSV):
        raise FileNotFoundError(
            f"Upload '{PROMPT_CSV}' to this Colab session first.\n"
            "In Colab: click the folder icon on the left → upload."
        )
    df = pd.read_csv(PROMPT_CSV, dtype=str).fillna("")
    # Coerce numeric columns
    df["id"]               = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["constraint_count"] = pd.to_numeric(df["constraint_count"], errors="coerce")
    df["steps"]            = pd.to_numeric(df["steps"], errors="coerce")
    print(f"Loaded {len(df)} prompts")
    for cat in ["instructability", "consistency", "planning"]:
        n = (df["category"] == cat).sum()
        print(f"  {cat}: {n}")
    return df
 


 
def collect_responses(prompts_df: pd.DataFrame) -> pd.DataFrame:
    save_path = f"{OUTPUT_DIR}all_responses.csv"
    fieldnames = ["id","category","prompt","difficulty","constraint_count",
                  "steps","paraphrase_group","source","model","response"]
 
    # Resume: load already-done (id, model) pairs
    done = set()
    if os.path.exists(save_path):
        existing = pd.read_csv(save_path, dtype=str)
        for _, r in existing.iterrows():
            done.add((str(r["id"]), r["model"]))
        print(f"Resuming — {len(done)} responses already saved.")
 
    # Open file in append mode
    file_exists = os.path.exists(save_path)
    fh = open(save_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    if not file_exists:
        writer.writeheader()
 
    total = len(prompts_df) * len(MODELS)
    pbar = tqdm(total=total, desc="Querying models")
 
    for _, row in prompts_df.iterrows():
        for model_name, model_id in MODELS.items():
            pbar.update(1)
            if (str(row["id"]), model_name) in done:
                continue
            response = query_model(model_id, str(row["prompt"]))
            record = {
                "id":               row["id"],
                "category":         row["category"],
                "prompt":           row["prompt"],
                "difficulty":       row["difficulty"],
                "constraint_count": row["constraint_count"],
                "steps":            row["steps"],
                "paraphrase_group": row["paraphrase_group"],
                "source":           row.get("source", ""),
                "model":            model_name,
                "response":         response,
            }
            writer.writerow(record)
            fh.flush()
            time.sleep(0.4)   # polite rate limit
 
    pbar.close()
    fh.close()
 
    result = pd.read_csv(save_path, dtype=str)
    result["constraint_count"] = pd.to_numeric(result["constraint_count"], errors="coerce")
    result["steps"]            = pd.to_numeric(result["steps"],            errors="coerce")
    print(f"\nTotal responses on disk: {len(result)}")
    return result

 
def extract_houses(text: str) -> set:
    t = text.lower()
    return {h for h in HOGWARTS if h in t}
 
def check_instructability(row) -> float:
    """
    Returns a score in {0.0, 0.4, 0.7, 1.0} based on:
      - Are the required houses present?
      - Are excluded houses absent?
      - Does the format match basic cues in the prompt?
    """
    prompt = str(row["prompt"]).lower()
    resp   = str(row["response"]).lower()
    if not resp:
        return 0.0
 
    # Determine excluded houses from prompt text
    excluded = set()
    for h in HOGWARTS:
        if h in prompt:
            idx = prompt.find(h)
            window = prompt[max(0, idx-40):idx+40]
            if any(kw in window for kw in ["exclud","except","not include","without","remov","no "]):
                excluded.add(h)
 
    expected = HOGWARTS - excluded
    found    = extract_houses(resp)
 
    content_ok  = expected.issubset(found)
    excluded_ok = not bool(excluded & found)
 
    # Basic format checks
    format_ok = True
    if "json"       in prompt: format_ok = "{" in resp or "[" in resp
    if "uppercase"  in prompt or "all caps" in prompt: format_ok = format_ok and any(c.isupper() for c in row["response"])
    if "comma"      in prompt and "separated" in prompt: format_ok = format_ok and "," in resp
    if "lowercase"  in prompt: format_ok = format_ok and row["response"] == row["response"].lower()
 
    if content_ok and excluded_ok and format_ok: return 1.0
    if content_ok and excluded_ok:               return 0.7
    if content_ok:                               return 0.4
    return 0.0
 
def check_planning(row) -> float:
    """
    Scores planning accuracy 0–1. Checks:
      - Non-empty response mentioning at least one house
      - Evidence of sequential operations in the response
      - Numerical output when the prompt asks for counts
      - Step-count alignment (more steps = higher bar)
    """
    prompt = str(row["prompt"]).lower()
    resp   = str(row["response"]).lower()
    if not resp or len(resp.split()) < 2:
        return 0.0
    if not extract_houses(resp):
        return 0.0
 
    score = 0.5
    ops = sum(1 for kw in ["sort","remov","filter","count","total","remain","result",
                            "step","output","then","first","second","final"] if kw in resp)
    if ops >= 2: score += 0.2
 
    if any(kw in prompt for kw in ["count","total","sum","how many"]):
        if re.search(r'\b\d+\b', resp): score += 0.15
 
    steps = row.get("steps")
    try:
        s = int(float(steps))
        if ops >= max(1, s - 1): score += 0.15
    except (TypeError, ValueError):
        pass
 
    return min(score, 1.0)
 
def tokenize(text: str) -> set:
    return set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
 
def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0
 

 
def compute_metrics(responses: pd.DataFrame) -> dict:
    R = {}
 
    # ── 8A. INSTRUCTABILITY ──────────────────────────────────
    inst = responses[responses["category"] == "instructability"].copy()
    inst["score"] = inst.apply(check_instructability, axis=1)
 
    # Overall per model
    R["inst_overall"] = inst.groupby("model")["score"].mean().round(4)
 
    # By constraint count (1,2,3,4)  ← KEY GRAPH 2
    inst["cc"] = inst["constraint_count"]
    R["inst_by_cc"] = (inst.groupby(["model","cc"])["score"]
                          .mean().reset_index()
                          .rename(columns={"cc":"constraint_count","score":"compliance"}))
 
    # By difficulty
    R["inst_by_diff"] = (inst.groupby(["model","difficulty"])["score"]
                            .mean().reset_index()
                            .rename(columns={"score":"compliance"}))
 
    # ── 8B. PLANNING ─────────────────────────────────────────
    plan = responses[responses["category"] == "planning"].copy()
    plan["score"] = plan.apply(check_planning, axis=1)
 
    # Overall per model
    R["plan_overall"] = plan.groupby("model")["score"].mean().round(4)
 
    # By step count (2,3,4,5,6,7)  ← KEY GRAPH 3
    plan["st"] = plan["steps"]
    R["plan_by_steps"] = (plan.groupby(["model","st"])["score"]
                             .mean().reset_index()
                             .rename(columns={"st":"steps","score":"accuracy"}))
 
    # ── 8C. CONSISTENCY (WITHIN-MODEL JACCARD) ───────────────
    cons = responses[responses["category"] == "consistency"].copy()
 
    within_records = []
    for model in cons["model"].unique():
        mdf = cons[cons["model"] == model]
        for grp, gdf in mdf.groupby("paraphrase_group"):
            if grp == "" or len(gdf) < 2: continue
            pairs = list(itertools.combinations(gdf["response"].tolist(), 2))
            scores = [jaccard(tokenize(a), tokenize(b)) for a, b in pairs]
            within_records.append({
                "model": model, "group": grp,
                "jaccard": np.mean(scores), "n_pairs": len(pairs)
            })
    R["within"] = pd.DataFrame(within_records)
    if not R["within"].empty:
        R["cons_overall"] = R["within"].groupby("model")["jaccard"].mean().round(4)
    else:
        R["cons_overall"] = pd.Series(dtype=float)
 
    # ── 8D. CROSS-MODEL JACCARD ──────────────────────────────
    cross_records = []
    for pid in cons["id"].unique():
        pdf = cons[cons["id"] == pid]
        mresp = {r["model"]: r["response"] for _, r in pdf.iterrows()}
        for (m1, r1), (m2, r2) in itertools.combinations(mresp.items(), 2):
            cross_records.append({
                "pair": f"{m1} – {m2}",
                "jaccard": jaccard(tokenize(r1), tokenize(r2))
            })
    R["cross"] = pd.DataFrame(cross_records)
 
    return R
 

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F8F9FA",
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
})
COLORS = [MODEL_COLORS.get(m, "#888") for m in MODELS]
 
def save(name):
    path = f"{OUTPUT_DIR}{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")
 
def plot_all(R: dict):
    model_list = list(MODELS.keys())
 
    # ── GRAPH 1: Overall bar chart, 3 metrics × 3 models ─────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("LLM Coherence Benchmark — Overall Performance", fontsize=14, fontweight="bold")
 
    datasets = [
        (R["inst_overall"],  "Instruction Compliance Rate"),
        (R["plan_overall"],  "Planning Accuracy"),
        (R["cons_overall"],  "Within-Model Consistency\n(Jaccard)"),
    ]
    for ax, (data, title) in zip(axes, datasets):
        if data.empty:
            ax.set_title(title); ax.text(0.5,0.5,"No data",ha="center",transform=ax.transAxes); continue
        vals   = [data.get(m, 0) for m in model_list]
        colors = [MODEL_COLORS.get(m,"#888") for m in model_list]
        bars   = ax.bar(model_list, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.tight_layout()
    save("graph1_overall_metrics")
 
    # ── GRAPH 2: Compliance vs. Constraint Count ──────────────
    cc_df = R["inst_by_cc"].dropna(subset=["constraint_count"])
    if not cc_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in model_list:
            sub = cc_df[cc_df["model"]==model].sort_values("constraint_count")
            if sub.empty: continue
            ax.plot(sub["constraint_count"], sub["compliance"],
                    marker="o", linewidth=2.5, markersize=9,
                    label=model, color=MODEL_COLORS.get(model,"#888"))
            # Annotate each point
            for _, row in sub.iterrows():
                ax.annotate(f"{row['compliance']:.2f}",
                            (row["constraint_count"], row["compliance"]),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=8, color=MODEL_COLORS.get(model,"#888"))
 
        ax.set_xlabel("Number of Constraints in Prompt", fontsize=12)
        ax.set_ylabel("Compliance Rate", fontsize=12)
        ax.set_title("Graph 2: Compliance Rate vs. Instruction Complexity\n"
                     "How fast does each model break under more rules?",
                     fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["1 Constraint\n(Easy)", "2 Constraints\n(Medium)",
                             "3 Constraints\n(Hard)", "4 Constraints\n(Very Hard)"])
        ax.legend(title="Model", fontsize=10)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        save("graph2_compliance_vs_constraints")
 
    # ── GRAPH 3: Planning Accuracy vs. Step Count ─────────────
    ps_df = R["plan_by_steps"].dropna(subset=["steps"])
    if not ps_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in model_list:
            sub = ps_df[ps_df["model"]==model].sort_values("steps")
            if sub.empty: continue
            ax.plot(sub["steps"], sub["accuracy"],
                    marker="s", linewidth=2.5, markersize=9,
                    label=model, color=MODEL_COLORS.get(model,"#888"))
            for _, row in sub.iterrows():
                ax.annotate(f"{row['accuracy']:.2f}",
                            (row["steps"], row["accuracy"]),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=8, color=MODEL_COLORS.get(model,"#888"))
 
        ax.set_xlabel("Number of Reasoning Steps Required", fontsize=12)
        ax.set_ylabel("Planning Accuracy", fontsize=12)
        ax.set_title("Graph 3: Planning Accuracy vs. Reasoning Chain Length\n"
                     "Do models lose track as tasks get longer?",
                     fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xticks(sorted(ps_df["steps"].dropna().unique()))
        ax.legend(title="Model", fontsize=10)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        save("graph3_planning_vs_steps")
 
    # ── GRAPH 4: Cross-Model Jaccard Heatmap ──────────────────
    if not R["cross"].empty:
        # Build symmetric matrix
        agg = R["cross"].groupby("pair")["jaccard"].mean()
        heat = pd.DataFrame(1.0, index=model_list, columns=model_list)
        for pair, val in agg.items():
            parts = [p.strip() for p in pair.split("–")]
            if len(parts) == 2 and parts[0] in heat.index and parts[1] in heat.columns:
                heat.loc[parts[0], parts[1]] = val
                heat.loc[parts[1], parts[0]] = val
 
        fig, ax = plt.subplots(figsize=(6, 5))
        mask = np.eye(len(model_list), dtype=bool)
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="Blues",
                    vmin=0.6, vmax=1.0, ax=ax, linewidths=1,
                    annot_kws={"size": 14, "weight": "bold"},
                    mask=~(~mask))          # show everything incl diagonal
        ax.set_title("Graph 4: Cross-Model Output Similarity\n"
                     "Jaccard Similarity of responses to identical prompts",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        save("graph4_cross_model_heatmap")
 
    # ── GRAPH 5: Within-model Consistency Boxplot ─────────────
    within = R["within"]
    if not within.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        data_per_model = [within[within["model"]==m]["jaccard"].values for m in model_list]
        bp = ax.boxplot(data_per_model, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        for patch, model in zip(bp["boxes"], model_list):
            patch.set_facecolor(MODEL_COLORS.get(model,"#888"))
            patch.set_alpha(0.65)
        ax.set_xticklabels(model_list, fontsize=11)
        ax.set_ylabel("Within-Group Jaccard Similarity", fontsize=12)
        ax.set_title("Graph 5: Consistency Distribution Across Paraphrase Groups\n"
                     "Narrower box = model gives more stable answers to same question rephrased",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        save("graph5_consistency_boxplot")
 
    # ── GRAPH 6: Radar / Spider Chart ─────────────────────────
    cc_df_r  = R["inst_by_cc"]
    ps_df_r  = R["plan_by_steps"]
    within_r = R["within"]
 
    dims = ["Compliance\n(1 constraint)", "Compliance\n(4 constraints)",
            "Planning\n(2 steps)", "Planning\n(5+ steps)", "Consistency"]
    N = len(dims)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
 
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    for model in model_list:
        try:
            c1 = cc_df_r[(cc_df_r["model"]==model)&(cc_df_r["constraint_count"]==1)]["compliance"].mean()
            c4 = cc_df_r[(cc_df_r["model"]==model)&(cc_df_r["constraint_count"]==4)]["compliance"].mean()
            p2 = ps_df_r[(ps_df_r["model"]==model)&(ps_df_r["steps"]==2)]["accuracy"].mean()
            p5 = ps_df_r[(ps_df_r["model"]==model)&(ps_df_r["steps"]>=5)]["accuracy"].mean()
            cj = within_r[within_r["model"]==model]["jaccard"].mean() if not within_r.empty else 0.5
            vals = [float(x) if not (x!=x) else 0.0 for x in [c1,c4,p2,p5,cj]]
        except Exception:
            vals = [0.0]*5
        vals += [vals[0]]
        col = MODEL_COLORS.get(model,"#888")
        ax.plot(angles, vals, linewidth=2.2, label=model, color=col)
        ax.fill(angles, vals, alpha=0.12, color=col)
 
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title("Graph 6: Multi-Dimensional Capability Radar\n"
                 "Larger area = better overall performance",
                 fontsize=12, fontweight="bold", pad=25)
    plt.tight_layout()
    save("graph6_radar")
 
    print("\n✅  All 6 graphs saved to results/")
 
# ─────────────────────────────────────────────────────────────
# CELL 10 — PRINT SUMMARY + WHAT TO COMPARE
# ─────────────────────────────────────────────────────────────
 
def print_summary(R: dict):
    sep = "─" * 60
    print(f"\n{sep}")
    print("  RESULTS SUMMARY")
    print(sep)
 
    print("\n📊  INSTRUCTION COMPLIANCE (overall)")
    print(R["inst_overall"].to_string())
 
    print("\n📊  COMPLIANCE BY CONSTRAINT COUNT")
    if not R["inst_by_cc"].empty:
        pt = R["inst_by_cc"].pivot_table(
                index="constraint_count", columns="model",
                values="compliance", aggfunc="mean").round(4)
        print(pt.to_string())
 
    print("\n📊  PLANNING ACCURACY (overall)")
    print(R["plan_overall"].to_string())
 
    print("\n📊  PLANNING ACCURACY BY STEP COUNT")
    if not R["plan_by_steps"].empty:
        pt = R["plan_by_steps"].pivot_table(
                index="steps", columns="model",
                values="accuracy", aggfunc="mean").round(4)
        print(pt.to_string())
 
    print("\n📊  WITHIN-MODEL CONSISTENCY (mean Jaccard per paraphrase group)")
    if not R["cons_overall"].empty:
        print(R["cons_overall"].to_string())
 
    print("\n📊  CROSS-MODEL JACCARD SIMILARITY")
    if not R["cross"].empty:
        print(R["cross"].groupby("pair")["jaccard"].mean().round(4).to_string())
 
    print(f"\n{sep}")
    print("  WHAT TO COMPARE IN YOUR PAPER")
    print(sep)
    print("""
Graph 2 — Compliance vs. Constraint Count:
  • Is there a measurable drop from constraint_count=1 → 4?
  • Which model degrades most / least?
  • Is the drop smooth (linear) or sudden (cliff at 3–4)?
  → Publishable claim: "Compliance degrades predictably with
    instruction density, with a steeper decline for [model X]."
 
Graph 3 — Planning Accuracy vs. Steps:
  • Does accuracy fall from step 2 → 7?
  • Is there a threshold (e.g., accuracy < 50% at 5+ steps)?
  • Which model best preserves accuracy at long chains?
  → Publishable claim: "Planning accuracy collapses beyond
    N steps, confirming LLMs are not reliable multi-step reasoners."
 
Graph 4 — Cross-Model Heatmap:
  • Is GPT–Gemini similarity always highest? (shared training?)
  • Does DeepSeek diverge more? (open-source vs proprietary?)
  → Insight: "High inter-model similarity suggests convergent
    representations, not independent reasoning."
 
Graph 5 — Consistency Boxplot:
  • Which model has the tightest IQR?
  • Are there outlier groups where a model collapses to ~0 Jaccard?
  → Claim: "[Model X] is the most reliably consistent paraphraser."
 
Graph 6 — Radar:
  • Is there a compliance–planning trade-off?
  • Which model is the strongest all-rounder?
 
Statistical test to add (optional but impressive):
  from scipy.stats import wilcoxon
  # Compare compliance at constraint_count=1 vs 4 for each model
  # p < 0.05 confirms degradation is statistically significant
""")
 
# ─────────────────────────────────────────────────────────────
# CELL 11 — OPTIONAL: STATISTICAL SIGNIFICANCE TEST
# ─────────────────────────────────────────────────────────────
 
def run_stat_tests(responses: pd.DataFrame):
    """
    Wilcoxon signed-rank test:
      H0: compliance at constraint_count=1 equals compliance at constraint_count=4
    Rejection (p<0.05) confirms complexity hurts performance.
    """
    inst = responses[responses["category"] == "instructability"].copy()
    inst["score"] = inst.apply(check_instructability, axis=1)
    inst["cc"]    = inst["constraint_count"]
 
    print("\n📊  WILCOXON TEST: constraint_count=1 vs 4")
    print("    (p < 0.05 = statistically significant degradation)")
    for model in inst["model"].unique():
        mdf = inst[inst["model"]==model]
        g1 = mdf[mdf["cc"]==1]["score"].values
        g4 = mdf[mdf["cc"]==4]["score"].values
        n  = min(len(g1), len(g4))
        if n < 10:
            print(f"  {model}: not enough data (n={n})")
            continue
        try:
            stat, p = stats.wilcoxon(g1[:n], g4[:n])
            sig = "✓ significant" if p < 0.05 else "✗ not significant"
            print(f"  {model}: p={p:.4f}  {sig}")
        except Exception as e:
            print(f"  {model}: test failed — {e}")
 
# ─────────────────────────────────────────────────────────────
# CELL 12 — MAIN PIPELINE  ← run this
# ─────────────────────────────────────────────────────────────
 
def main():
    print("=" * 60)
    print("  LLM COHERENCE BENCHMARK — STARTING PIPELINE")
    print("=" * 60)
 
    # 1. Load prompts (reads metadata directly from CSV)
    prompts_df = load_prompts()
 
    # 2. Query models and save responses (resumable)
    print(f"\nQuerying {len(MODELS)} models × {len(prompts_df)} prompts…")
    print("Responses saved incrementally — safe to interrupt and resume.\n")
    responses = collect_responses(prompts_df)
 
    # 3. Compute metrics
    print("\nComputing metrics…")
    R = compute_metrics(responses)
 
    # 4. Print summary + comparison guide
    print_summary(R)
 
    # 5. Plot all 6 graphs
    print("\nGenerating graphs…")
    plot_all(R)
 
    # 6. Statistical tests
    run_stat_tests(responses)
 
    # 7. Save scored CSVs for paper
    responses_path = f"{OUTPUT_DIR}responses_scored.csv"
    responses.to_csv(responses_path, index=False)
    R["inst_by_cc"].to_csv(f"{OUTPUT_DIR}compliance_by_constraints.csv",   index=False)
    R["plan_by_steps"].to_csv(f"{OUTPUT_DIR}planning_by_steps.csv",         index=False)
    if not R["cross"].empty:
        R["cross"].to_csv(f"{OUTPUT_DIR}cross_model_jaccard.csv",           index=False)
    if not R["within"].empty:
        R["within"].to_csv(f"{OUTPUT_DIR}within_model_jaccard.csv",         index=False)
 
    print(f"\n✅  Pipeline complete. All outputs in '{OUTPUT_DIR}'")
 
# ─── RUN ────────────────────────────────────────────────────
main()
 
