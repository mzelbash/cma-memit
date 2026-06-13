"""
Causal Mediation Analysis and MEMIT
SEAS 8525 - Computer Vision and Generative AI
Dr. Elbasheer
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

st.set_page_config(
    page_title="CMA & MEMIT",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; color:#0f172a; }

.sidebar-brand   { padding:16px 0 20px; border-bottom:1px solid #e2e8f0; margin-bottom:14px; }
.sidebar-title   { font-size:1.10rem; font-weight:800; color:#1e1b4b; letter-spacing:-0.02em; }
.sidebar-course  { font-size:0.88rem; font-weight:700; color:#3730a3; margin-top:4px; }
.sidebar-course-name { font-size:0.82rem; font-weight:600; color:#4f46e5; }
.sidebar-instructor  { font-size:0.77rem; color:#6366f1; font-style:italic; }

.section-hdr { padding:22px 28px 18px; border-radius:12px;
               background:linear-gradient(135deg,#eef2ff 0%,#f8fafc 100%);
               border-left:5px solid #4f46e5; margin-bottom:24px; }
.section-num  { font-size:0.72rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.1em; color:#6366f1; margin-bottom:4px; }
.section-title { font-size:1.50rem; font-weight:800; color:#1e1b4b;
                 letter-spacing:-0.02em; line-height:1.2; }
.section-sub   { font-size:0.88rem; color:#475569; margin-top:5px; }

.info-box    { background:#eef2ff; border-left:4px solid #6366f1; border-radius:6px;
               padding:12px 16px; margin:12px 0; font-size:0.875rem; color:#1e1b4b; }
.warn-box    { background:#fffbeb; border-left:4px solid #f59e0b; border-radius:6px;
               padding:12px 16px; margin:12px 0; font-size:0.875rem; color:#1e1b4b; }
.success-box { background:#f0fdf4; border-left:4px solid #22c55e; border-radius:6px;
               padding:12px 16px; margin:12px 0; font-size:0.875rem; color:#1e1b4b; }
.class-tip   { background:#fdf4ff; border-left:4px solid #a855f7; border-radius:6px;
               padding:12px 16px; margin:12px 0; font-size:0.875rem; color:#1e1b4b; }
.class-tip strong { color:#7e22ce; }

.enc-cards { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:14px 0 20px; }
.enc-card  { border-radius:9px; padding:14px 16px; }
.enc-card-title { font-size:0.88rem; font-weight:700; margin-bottom:7px; }
.enc-card-body  { font-size:0.82rem; color:#374151; line-height:1.68; }

.math-box  { background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #4f46e5;
             border-radius:6px; padding:14px 18px; margin:10px 0; }
.math-step { display:flex; gap:10px; margin-bottom:10px; align-items:flex-start; }
.math-step:last-child { margin-bottom:0; }
.math-step-num { background:#4f46e5; color:#fff; border-radius:50%; width:22px; height:22px;
                 font-size:0.70rem; display:flex; align-items:center; justify-content:center;
                 flex-shrink:0; font-weight:700; }

.run-box { border-radius:10px; padding:14px 18px; margin-bottom:12px; }
.run-clean   { background:#f0fdf4; border:2px solid #16a34a; }
.run-corrupt { background:#fef2f2; border:2px solid #dc2626; }
.run-patch   { background:#eef2ff; border:2px solid #4f46e5; }
.run-label   { font-size:0.72rem; font-weight:700; text-transform:uppercase;
               letter-spacing:0.08em; margin-bottom:6px; }
.run-label.clean   { color:#14532d; }
.run-label.corrupt { color:#991b1b; }
.run-label.patch   { color:#3730a3; }
.run-body { font-size:0.82rem; color:#374151; line-height:1.6; }

.kv-row   { display:flex; align-items:center; gap:8px; margin:6px 0;
            font-family:'JetBrains Mono',monospace; font-size:0.80rem; }
.kv-key   { background:#dbeafe; color:#1d4ed8; border-radius:5px;
            padding:4px 10px; font-weight:600; }
.kv-arrow { color:#6366f1; font-size:1.1rem; font-weight:700; }
.kv-val   { background:#d1fae5; color:#065f46; border-radius:5px;
            padding:4px 10px; font-weight:600; }
.kv-val-new { background:#fef3c7; color:#92400e; border-radius:5px;
              padding:4px 10px; font-weight:600; }

.edit-before { background:#fef2f2; border:1px solid #fca5a5; border-radius:6px;
               padding:10px 14px; font-size:0.85rem; margin-bottom:6px; }
.edit-after  { background:#f0fdf4; border:1px solid #86efac; border-radius:6px;
               padding:10px 14px; font-size:0.85rem; }
.edit-label  { font-size:0.68rem; font-weight:700; text-transform:uppercase;
               letter-spacing:0.07em; margin-bottom:4px; }

.comp-table { width:100%; border-collapse:collapse; font-family:'Inter',sans-serif;
              font-size:0.875rem; margin:0.5rem 0 1.2rem; }
.comp-table thead th { background:#f8fafc; color:#374151; font-weight:700;
    font-size:0.70rem; letter-spacing:0.07em; text-transform:uppercase;
    padding:11px 16px; text-align:left; border-bottom:2px solid #e2e8f0; }
.comp-table tbody td { padding:10px 16px; color:#1e1b4b; border-bottom:1px solid #f1f5f9;
    vertical-align:top; line-height:1.55; }
.comp-table tbody tr:last-child td { border-bottom:none; }
.comp-table tbody tr:hover td { background:#fafbff; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def section_header(num, title, subtitle):
    st.markdown(f"""
    <div class="section-hdr">
      <div class="section-num">Section {num}</div>
      <div class="section-title">{title}</div>
      <div class="section-sub">{subtitle}</div>
    </div>""", unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def tip(text):
    st.markdown(f'<div class="class-tip"><strong>Class Tip:</strong> {text}</div>',
                unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-title">CMA &amp; MEMIT</div>
      <div class="sidebar-course">SEAS 8525</div>
      <div class="sidebar-course-name">Computer Vision &amp; Generative AI</div>
      <div class="sidebar-instructor">Dr. Elbasheer</div>
    </div>""", unsafe_allow_html=True)

    sections = [
        "1. Where Do Transformers Store Facts?",
        "2. Causal Mediation Analysis",
        "3. MEMIT: Editing a Stored Fact",
        "4. Why It Matters + Tools",
    ]
    section = st.radio("Sections", sections, label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top:20px;padding-top:14px;border-top:1px solid #e2e8f0;">
      <div style="font-size:0.70rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#94a3b8;margin-bottom:8px;">
        Running example
      </div>
      <div style="background:#fef3c7;border-radius:7px;padding:10px 12px;
                  font-size:0.80rem;color:#92400e;font-weight:600;
                  font-family:'JetBrains Mono',monospace;">
        "The capital of France is ___"
      </div>
      <div style="font-size:0.75rem;color:#64748b;margin-top:8px;line-height:1.6;">
        Used throughout all sections to ground every concept in one concrete example.
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: WHERE DO TRANSFORMERS STORE FACTS?
# ══════════════════════════════════════════════════════════════════════════════
if section.startswith("1"):
    section_header("1", "Where Do Transformers Store Facts?",
                   "MLP layers act as key-value memories: the foundation of CMA and MEMIT")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    When a language model completes "The capital of France is ___" with "Paris",
    that answer is not magic. It is stored somewhere in the model's 12 billion parameters.
    The question is: <strong>exactly where, and in what form?</strong>
    Research (Geva et al. 2021, Meng et al. 2022) found a clear answer:
    factual associations live primarily in the <strong>MLP layers</strong> of the middle-to-late
    transformer blocks, encoded as implicit key-value pairs.
    </p>
    """, unsafe_allow_html=True)

    # Attention vs MLP intuition
    st.markdown("### Attention vs. MLP: two different jobs")
    st.markdown("""
    <div class="enc-cards">
      <div class="enc-card" style="background:#dbeafe;border-left:4px solid #2563eb;">
        <div class="enc-card-title" style="color:#1d4ed8;">Attention layers: routing information</div>
        <div class="enc-card-body">
          Attention heads decide <strong>which tokens to look at</strong> and move information
          between positions. When processing "The capital of France", attention heads
          route the representation of "France" to the position where the prediction
          will be made.<br><br>
          Think of attention as: <em>"gather the relevant context."</em>
        </div>
      </div>
      <div class="enc-card" style="background:#d1fae5;border-left:4px solid #059669;">
        <div class="enc-card-title" style="color:#065f46;">MLP layers: storing facts</div>
        <div class="enc-card-body">
          MLP layers act as <strong>factual memory</strong>. Given the enriched representation
          of "France" (after attention has gathered context), the MLP retrieves
          the associated fact: "Paris".<br><br>
          Think of MLP as: <em>"look up what I know about this."</em>
          Geva et al. showed each MLP neuron fires for a specific concept or fact.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Key-value memory view
    st.markdown("### The MLP as a key-value memory")
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:12px;'>
    A two-layer MLP with hidden size 4h and input size h performs exactly two matrix multiplications.
    Researchers noticed this maps perfectly onto a key-value lookup:
    </p>
    """, unsafe_allow_html=True)

    col_formula, col_diagram = st.columns([1, 1])
    with col_formula:
        st.markdown("""
        <div class="math-box">
          <div class="math-step">
            <div class="math-step-num">1</div>
            <div>
              <strong>Keys</strong> = rows of W<sub>1</sub> (the up-projection matrix).<br>
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;">
              k<sub>i</sub> = W<sub>1</sub>[i, :]
              </span><br>
              <span style="font-size:0.78rem;color:#475569;">
              Each row is a 768-dim key vector associated with one "memory slot."
              </span>
            </div>
          </div>
          <div class="math-step">
            <div class="math-step-num">2</div>
            <div>
              <strong>Activation scores</strong> = how much the input matches each key.<br>
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;">
              m = GELU(W<sub>1</sub> h) &nbsp; shape: (4h,)
              </span><br>
              <span style="font-size:0.78rem;color:#475569;">
              High score = the input h is similar to that key row.
              </span>
            </div>
          </div>
          <div class="math-step">
            <div class="math-step-num">3</div>
            <div>
              <strong>Values</strong> = columns of W<sub>2</sub> (the down-projection matrix).<br>
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;">
              v<sub>i</sub> = W<sub>2</sub>[:, i]
              </span><br>
              <span style="font-size:0.78rem;color:#475569;">
              Each column is a 768-dim value vector encoding a factual concept.
              </span>
            </div>
          </div>
          <div class="math-step">
            <div class="math-step-num">4</div>
            <div>
              <strong>MLP output</strong> = weighted sum of values.<br>
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;">
              out = W<sub>2</sub> m = &sum;<sub>i</sub> m<sub>i</sub> v<sub>i</sub>
              </span><br>
              <span style="font-size:0.78rem;color:#475569;">
              The output is dominated by the values whose keys matched the input.
              </span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_diagram:
        # Draw MLP as key-value memory
        fig, ax = plt.subplots(figsize=(5.5, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off")

        def box(x, y, w, h, fc, ec, text, fs=9, tc="white", bold=False):
            ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0.15", facecolor=fc, edgecolor=ec, linewidth=1.5))
            ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                    fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
                    fontfamily="monospace")

        # Input
        box(0.3, 4.2, 2.2, 1.6, "#dbeafe", "#2563eb", "h\n(768)", tc="#1d4ed8", bold=True)
        ax.annotate("", xy=(3.0, 5.0), xytext=(2.5, 5.0),
                    arrowprops=dict(arrowstyle="->", color="#4f46e5", lw=1.8))

        # W1 keys
        for i, (label, fc, ec) in enumerate([
            ("k₁: France?",  "#fef3c7", "#d97706"),
            ("k₂: Germany?", "#f1f5f9", "#94a3b8"),
            ("k₃: Italy?",   "#f1f5f9", "#94a3b8"),
            ("k₄: Spain?",   "#f1f5f9", "#94a3b8"),
        ]):
            y = 7.8 - i * 1.85
            box(3.0, y, 2.4, 1.4, fc, ec, label, fs=8,
                tc="#92400e" if i == 0 else "#64748b")

        ax.text(4.2, 9.6, "W₁ rows = Keys", ha="center", fontsize=8,
                color="#374151", fontweight="bold")

        # Activation arrows
        for i, (w, col) in enumerate([(0.95, "#d97706"), (0.15, "#94a3b8"),
                                       (0.05, "#94a3b8"), (0.02, "#94a3b8")]):
            y = 8.5 - i * 1.85
            ax.annotate("", xy=(6.0, y), xytext=(5.4, y),
                        arrowprops=dict(arrowstyle="->",
                                        color=col, lw=1 + w * 3))

        # W2 values
        for i, (label, fc, ec) in enumerate([
            ("v₁: Paris",   "#d1fae5", "#059669"),
            ("v₂: Berlin",  "#f1f5f9", "#94a3b8"),
            ("v₃: Rome",    "#f1f5f9", "#94a3b8"),
            ("v₄: Madrid",  "#f1f5f9", "#94a3b8"),
        ]):
            y = 7.8 - i * 1.85
            box(6.0, y, 2.4, 1.4, fc, ec, label, fs=8,
                tc="#065f46" if i == 0 else "#64748b")

        ax.text(7.2, 9.6, "W₂ cols = Values", ha="center", fontsize=8,
                color="#374151", fontweight="bold")

        # Output arrow
        ax.annotate("", xy=(9.5, 5.0), xytext=(8.4, 5.0),
                    arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.0))
        ax.text(9.7, 5.0, "Paris", ha="left", va="center",
                fontsize=10, color="#065f46", fontweight="bold",
                fontfamily="monospace")

        ax.text(5.0, 0.3,
                "Input 'France' activates key k₁ strongly\n"
                "→ value v₁ (Paris) dominates the output",
                ha="center", fontsize=8, color="#374151",
                style="italic")

        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    tip("""This key-value view is an <em>interpretation</em>, not a literal database.
    The MLP does not have labeled slots. But the math works out to be equivalent,
    and this framing directly motivated MEMIT's weight-editing approach:
    if facts are stored as (key, value) pairs in W<sub>1</sub> and W<sub>2</sub>,
    we can update those pairs surgically without touching anything else.""")

    st.divider()

    # Which layers store what
    st.markdown("### Which layers store which kinds of knowledge?")

    layers = list(range(1, 13))
    np.random.seed(42)
    factual   = np.array([0.05,0.08,0.12,0.18,0.35,0.62,0.80,0.88,0.85,0.72,0.55,0.38])
    syntactic = np.array([0.82,0.78,0.70,0.60,0.45,0.30,0.22,0.18,0.15,0.14,0.12,0.11])
    positional= np.array([0.75,0.65,0.50,0.35,0.22,0.15,0.12,0.10,0.09,0.08,0.07,0.06])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(layers, factual,    "o-", color="#4f46e5", lw=2.2, ms=6, label="Factual knowledge (e.g., capitals)")
    ax.plot(layers, syntactic,  "s-", color="#d97706", lw=2.2, ms=6, label="Syntactic structure (e.g., grammar)")
    ax.plot(layers, positional, "^-", color="#16a34a", lw=2.2, ms=6, label="Positional / surface patterns")
    ax.axvspan(5.5, 9.5, alpha=0.08, color="#4f46e5", label="Primary factual storage zone")
    ax.set_xlabel("Transformer layer (1 = earliest)", color="#475569", fontsize=9)
    ax.set_ylabel("Relative importance", color="#475569", fontsize=9)
    ax.set_xticks(layers)
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.spines[:].set_color("#e2e8f0")
    ax.legend(fontsize=8, framealpha=0.9, loc="center right")
    ax.set_title("Schematic: what different layers tend to encode (based on Geva et al. 2021)",
                 color="#0f172a", fontsize=9, pad=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    info("""The chart above is a schematic based on research findings, not exact numbers.
    The key takeaway: early layers handle syntax and position;
    middle-to-late layers handle factual associations.
    CMA (Section 2) makes this precise by measuring it on a real model.""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CAUSAL MEDIATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("2"):
    section_header("2", "Causal Mediation Analysis",
                   "A systematic experiment to find exactly which component stores a given fact")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    Section 1 gave the intuition: facts live in MLP layers. But <em>which</em> layers,
    exactly? And for <em>this specific fact</em> ("France" maps to "Paris")?
    Causal Mediation Analysis answers this with a controlled experiment:
    corrupt the input, then restore one component at a time and see what happens.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("### The three-run protocol")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class="run-box run-clean">
          <div class="run-label clean">Run 1: Clean run</div>
          <div class="run-body">
            Feed the model the <strong>original prompt</strong>:
            <br><br>
            <span style="font-family:'JetBrains Mono',monospace;">
            "The capital of France is"
            </span>
            <br><br>
            Record and save <strong>every activation</strong> at every layer
            and every component (attention heads, MLP outputs, residual stream).
            <br><br>
            Output: <strong style="color:#14532d;">"Paris"</strong> (correct)
          </div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="run-box run-corrupt">
          <div class="run-label corrupt">Run 2: Corrupted run</div>
          <div class="run-body">
            Feed the model a <strong>corrupted version</strong> of the prompt:
            <br><br>
            <span style="font-family:'JetBrains Mono',monospace;">
            "The capital of [noise] is"
            </span>
            <br><br>
            The subject token "France" is replaced with noise or a different word.
            The model no longer has the information it needs.
            <br><br>
            Output: <strong style="color:#991b1b;">"unknown" / wrong</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="run-box run-patch">
          <div class="run-label patch">Run 3: Patched run (repeated per component)</div>
          <div class="run-body">
            Start from the corrupted run, but <strong>restore one component</strong>
            from the clean run's saved activations.
            <br><br>
            Run this once for every layer's MLP, every attention head, etc.
            <br><br>
            If restoring component X makes the output go back to
            <strong style="color:#3730a3;">"Paris"</strong>,
            then X <strong>causally mediates</strong> the fact.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Causal indirect effect interactive simulation
    st.markdown("### Interactive simulation: Causal Indirect Effect per layer")
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:12px;'>
    The <strong>Causal Indirect Effect (CIE)</strong> measures how much restoring
    a component increases the probability of the correct token ("Paris").
    A high CIE means that component stores the fact.
    Adjust the sliders below to explore how the signal shifts across layers for different
    types of knowledge.
    </p>
    """, unsafe_allow_html=True)

    col_ctrl, col_chart = st.columns([1, 2])
    with col_ctrl:
        fact_type = st.selectbox(
            "Fact type to analyze",
            ["Capital city (France -> Paris)",
             "Inventor (telephone -> Bell)",
             "Author (Hamlet -> Shakespeare)",
             "Element symbol (Gold -> Au)"],
            index=0
        )
        num_layers = st.slider("Number of transformer layers", 6, 24, 12, step=6)
        show_attn  = st.checkbox("Show attention head CIE", value=True)
        show_mlp   = st.checkbox("Show MLP CIE", value=True)

    # Simulate CIE curves based on fact type
    np.random.seed({"Capital city (France -> Paris)": 0,
                    "Inventor (telephone -> Bell)": 1,
                    "Author (Hamlet -> Shakespeare)": 2,
                    "Element symbol (Gold -> Au)": 3}[fact_type])

    layers = np.arange(1, num_layers + 1)
    peak_map = {
        "Capital city (France -> Paris)": num_layers * 0.62,
        "Inventor (telephone -> Bell)":   num_layers * 0.70,
        "Author (Hamlet -> Shakespeare)": num_layers * 0.55,
        "Element symbol (Gold -> Au)":    num_layers * 0.58,
    }
    peak = peak_map[fact_type]

    def gaussian(x, mu, sigma, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    mlp_cie  = (gaussian(layers, peak, num_layers*0.12, 0.82)
                + np.random.normal(0, 0.03, num_layers)).clip(0, 1)
    attn_cie = (gaussian(layers, peak*0.45, num_layers*0.15, 0.28)
                + gaussian(layers, peak*0.85, num_layers*0.10, 0.18)
                + np.random.normal(0, 0.02, num_layers)).clip(0, 1)

    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        bar_w = 0.38
        x = np.arange(num_layers)
        if show_mlp:
            bars = ax.bar(x - bar_w/2, mlp_cie, width=bar_w,
                          color="#4f46e5", alpha=0.85, label="MLP CIE",
                          edgecolor="white", linewidth=0.5)
            peak_idx = np.argmax(mlp_cie)
            bars[peak_idx].set_edgecolor("#1e1b4b")
            bars[peak_idx].set_linewidth(2.5)
        if show_attn:
            ax.bar(x + bar_w/2, attn_cie, width=bar_w,
                   color="#d97706", alpha=0.75, label="Attention CIE",
                   edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Layer", color="#475569", fontsize=9)
        ax.set_ylabel("Causal Indirect Effect", color="#475569", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.spines[:].set_color("#e2e8f0")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.set_title(f"CIE per layer: \"{fact_type}\"",
                     color="#0f172a", fontsize=10, pad=8)

        if show_mlp:
            peak_layer = layers[np.argmax(mlp_cie)]
            ax.annotate(f"Peak: layer {peak_layer}",
                        xy=(peak_layer - 1, mlp_cie[peak_layer - 1]),
                        xytext=(peak_layer - 1 + 1.2, mlp_cie[peak_layer-1] + 0.08),
                        fontsize=8, color="#3730a3",
                        arrowprops=dict(arrowstyle="->", color="#6366f1", lw=1.2))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.82rem;color:#475569;margin-top:4px;'>
    <strong>Reading the chart:</strong> A tall MLP bar at layer L means that restoring
    the MLP output at layer L (from the clean run) brings the model's prediction
    back to the correct answer. That layer is where the fact is causally stored.
    Attention CIE is typically smaller and appears at earlier layers.
    </p>
    """, unsafe_allow_html=True)

    tip("""CMA does not need any labels or fine-tuning. It only requires running the model
    three times per component. For a 12-layer model with MLP + 12 attention heads per layer,
    that is about 12 * 13 = 156 forward passes per fact.
    This is exactly the method used in the ROME and MEMIT papers to identify layers 5-8
    as the primary storage location in GPT-2 and layers 13-17 in GPT-J.""")

    st.divider()

    with st.expander("Show the activation patching code (PyTorch)", expanded=False):
        st.code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name)

clean_prompt   = "The capital of France is"
corrupt_prompt = "The capital of zzzzz is"   # corrupted subject

# Step 1: clean run -- save all MLP outputs
clean_activations = {}
hooks = []
for layer_idx, layer in enumerate(model.transformer.h):
    def make_hook(idx):
        def hook(module, input, output):
            clean_activations[idx] = output[0].detach().clone()
        return hook
    hooks.append(layer.mlp.register_forward_hook(make_hook(layer_idx)))

with torch.no_grad():
    clean_ids  = tokenizer(clean_prompt, return_tensors="pt").input_ids
    clean_out  = model(clean_ids)
    clean_prob = clean_out.logits[0, -1].softmax(-1)

for h in hooks:
    h.remove()

# Step 2: patch each layer -- measure how much CIE is restored
target_id = tokenizer(" Paris", return_tensors="pt").input_ids[0, 0]
cie_per_layer = []

for patch_layer in range(len(model.transformer.h)):
    patch_hooks = []

    def make_patch(idx, saved):
        def hook(module, input, output):
            if idx == patch_layer:
                return (saved,) + output[1:]
            return output
        return hook

    for idx, layer in enumerate(model.transformer.h):
        patch_hooks.append(
            layer.mlp.register_forward_hook(make_patch(idx, clean_activations[idx]))
        )

    with torch.no_grad():
        corrupt_ids  = tokenizer(corrupt_prompt, return_tensors="pt").input_ids
        patched_out  = model(corrupt_ids)
        patched_prob = patched_out.logits[0, -1].softmax(-1)

    for h in patch_hooks:
        h.remove()

    cie_per_layer.append(patched_prob[target_id].item())

print("CIE per MLP layer:", [f"{v:.3f}" for v in cie_per_layer])
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MEMIT
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("3"):
    section_header("3", "MEMIT: Editing a Stored Fact",
                   "A closed-form weight update that rewrites the key-value pair in the MLP")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    CMA told us that the fact "France capital Paris" is stored in layers 5-8.
    MEMIT (<em>Mass-Editing Memory In a Transformer</em>, Meng et al. 2022) uses that
    information to update the MLP weights in those layers directly,
    replacing the old key-value pair with a new one.
    No retraining. No gradient descent. A single closed-form matrix update.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("### The four-step edit process")

    st.markdown("""
    <div class="math-box">
      <div class="math-step">
        <div class="math-step-num">1</div>
        <div>
          <strong>Extract the key vector</strong> for the subject to be edited.<br>
          Run the clean model on the subject tokens ("The capital of France is")
          and record the hidden state <strong>h</strong> at the identified MLP layer.
          This is the key: it is what the MLP uses to look up "France."<br>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;
                       color:#1d4ed8;">k = h &isin; &Ropf;<sup>768</sup></span>
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">2</div>
        <div>
          <strong>Compute the target value vector</strong> for the new fact.<br>
          What should the MLP output when it sees this key?
          Run the model with the desired continuation ("Lyon") and extract the
          residual stream vector that would produce that token.
          This is the target value <strong>v*</strong>.<br>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;
                       color:#065f46;">v* &isin; &Ropf;<sup>768</sup> &nbsp; (encodes "Lyon")</span>
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">3</div>
        <div>
          <strong>Solve for the weight update</strong> using a closed-form formula.<br>
          We want the new weight matrix W* to satisfy W* k = v*
          while changing as little as possible elsewhere.
          The solution is the rank-1 (or low-rank) update:<br>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;color:#7e22ce;">
          &Delta;W = (v* &minus; Wk) k<sup>T</sup> / (k<sup>T</sup> C<sup>&minus;1</sup> k)
          </span><br>
          <span style="font-size:0.78rem;color:#475569;">
          C is a covariance matrix of typical activations (precomputed once).
          It ensures the update respects the geometry of the weight space.
          </span>
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">4</div>
        <div>
          <strong>Apply the update</strong> to the MLP's second weight matrix W<sub>2</sub>.<br>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.80rem;color:#92400e;">
          W<sub>2</sub> &larr; W<sub>2</sub> + &Delta;W
          </span><br>
          The model is now updated. No gradient, no optimizer, no training loop.
          For mass editing (many facts at once), MEMIT batches the updates across
          multiple layers to spread the edits and reduce interference.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Interactive edit demo
    st.markdown("### Interactive demo: simulate a fact edit")
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:12px;'>
    Enter a subject, the current answer the model knows, and the new answer you want to inject.
    The simulation below computes the weight update numerically and shows what changes.
    </p>
    """, unsafe_allow_html=True)

    col_edit1, col_edit2, col_edit3 = st.columns(3)
    with col_edit1:
        subject   = st.text_input("Subject (prompt ending)", value="The capital of France is")
    with col_edit2:
        old_fact  = st.text_input("Current answer (what model knows)", value="Paris")
    with col_edit3:
        new_fact  = st.text_input("New answer (what to inject)", value="Lyon")

    if st.button("Simulate MEMIT weight update", type="primary"):
        np.random.seed(42)
        D = 768

        # Simulate key vector for the subject
        k = np.random.randn(D); k /= np.linalg.norm(k)

        # Simulate old and new value vectors
        v_old = np.random.randn(D); v_old /= np.linalg.norm(v_old)
        v_new = v_old + np.random.randn(D) * 0.4
        v_new /= np.linalg.norm(v_new)

        # Simulate W2 (the MLP down-projection)
        W2 = np.random.randn(D, D) * 0.02

        # Compute delta W (rank-1 MEMIT update, simplified: C = I)
        residual = (v_new - W2 @ k)
        dW = np.outer(residual, k) / (k @ k)

        W2_new = W2 + dW

        # Measure effects
        old_output_norm = np.linalg.norm(W2 @ k)
        new_output_norm = np.linalg.norm(W2_new @ k)
        delta_norm      = np.linalg.norm(dW)
        relative_change = delta_norm / np.linalg.norm(W2) * 100

        # Show results
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(f"""
            <div class="edit-before">
              <div class="edit-label" style="color:#991b1b;">Before edit</div>
              <strong>Prompt:</strong> {subject}<br>
              <strong>Model answers:</strong>
              <span style="color:#991b1b;font-weight:700;">{old_fact}</span>
            </div>
            <div class="edit-after">
              <div class="edit-label" style="color:#14532d;">After edit</div>
              <strong>Prompt:</strong> {subject}<br>
              <strong>Model answers:</strong>
              <span style="color:#14532d;font-weight:700;">{new_fact}</span>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            st.markdown("**Update statistics:**")
            m1, m2 = st.columns(2)
            m1.metric("||k|| (key norm)", f"{np.linalg.norm(k):.3f}")
            m2.metric("||v* - Wk||", f"{np.linalg.norm(residual):.3f}")
            m1.metric("||ΔW||", f"{delta_norm:.4f}")
            m2.metric("Weight change", f"{relative_change:.3f}%")
            success(f"""Weight update applied to W<sub>2</sub> at the identified MLP layer.
            Only <strong>{relative_change:.3f}%</strong> of the total weight matrix changed.
            Facts not involving "{subject.split()[-2] if len(subject.split()) > 2 else 'this subject'}"
            are unaffected.""")

        # Visualize delta W as a heatmap (subsampled)
        st.markdown("**Weight update matrix ΔW (first 64×64 values):**")
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.78rem;color:#475569;"
            "margin-bottom:6px;'>The rank-1 structure is clearly visible: the update "
            "is the outer product of two vectors, creating a low-rank stripe pattern.</p>",
            unsafe_allow_html=True
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        fig.patch.set_facecolor("white")

        vmax = np.abs(dW[:64, :64]).max()
        im1 = axes[0].imshow(dW[:64, :64], cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="auto")
        axes[0].set_title("ΔW (rank-1 update, 64x64 submatrix)",
                          color="#0f172a", fontsize=9, pad=6)
        axes[0].set_xlabel("Column index", color="#475569", fontsize=8)
        axes[0].set_ylabel("Row index", color="#475569", fontsize=8)
        axes[0].tick_params(colors="#64748b", labelsize=7)
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Show singular values to confirm rank-1
        U, S, Vt = np.linalg.svd(dW, full_matrices=False)
        axes[1].bar(range(1, 16), S[:15], color=["#4f46e5"] + ["#c7d2fe"]*14,
                    edgecolor="white")
        axes[1].set_title("Singular values of ΔW\n(rank-1: only the first is non-zero)",
                          color="#0f172a", fontsize=9, pad=6)
        axes[1].set_xlabel("Singular value index", color="#475569", fontsize=8)
        axes[1].set_ylabel("Magnitude", color="#475569", fontsize=8)
        axes[1].tick_params(colors="#64748b", labelsize=8)
        axes[1].spines[:].set_color("#e2e8f0")
        axes[1].set_facecolor("white")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    else:
        info("Fill in the subject, old answer, and new answer, then click the button.")

    st.divider()

    st.markdown("### What makes MEMIT different from ROME?")
    st.markdown("""
    <table class="comp-table" style="table-layout:fixed;width:100%;">
      <colgroup><col style="width:18%;"><col style="width:41%;"><col style="width:41%;"></colgroup>
      <thead>
        <tr><th>Aspect</th><th>ROME (2022)</th><th>MEMIT (2022)</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Scale</strong></td>
          <td>One fact at a time</td>
          <td>Hundreds or thousands of facts in a single batch</td>
        </tr>
        <tr>
          <td><strong>Layers updated</strong></td>
          <td>Single layer (the peak CIE layer)</td>
          <td>Multiple layers simultaneously; edits distributed to reduce interference</td>
        </tr>
        <tr>
          <td><strong>Update formula</strong></td>
          <td>Rank-1 update to one layer's W<sub>2</sub></td>
          <td>Distributed low-rank updates across L layers, with least-squares constraint</td>
        </tr>
        <tr>
          <td><strong>Side effects</strong></td>
          <td>Can degrade unrelated knowledge for large edits</td>
          <td>Lower side effects; distributing edits reduces each layer's individual load</td>
        </tr>
        <tr>
          <td><strong>Use case</strong></td>
          <td>Single targeted correction (e.g., fix one wrong fact)</td>
          <td>Dataset-scale knowledge updates (e.g., update 10,000 outdated facts)</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: WHY IT MATTERS + TOOLS
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("4"):
    section_header("4", "Why It Matters + Tools",
                   "Real-world applications and the libraries that implement CMA and MEMIT")

    st.markdown("### Why model editing matters")
    st.markdown("""
    <div class="enc-cards">
      <div class="enc-card" style="background:#dbeafe;border-left:4px solid #2563eb;">
        <div class="enc-card-title" style="color:#1d4ed8;">Correcting outdated knowledge</div>
        <div class="enc-card-body">
          A model trained in 2022 may believe the wrong person holds a political office
          or that a company has an old CEO. Full retraining is prohibitively expensive.
          MEMIT can update thousands of outdated facts in minutes, producing a model
          that reflects current reality.
        </div>
      </div>
      <div class="enc-card" style="background:#fef2f2;border-left:4px solid #dc2626;">
        <div class="enc-card-title" style="color:#991b1b;">Privacy and data removal</div>
        <div class="enc-card-body">
          A model trained on public data may have memorized personal information
          (phone numbers, addresses, medical records). CMA can locate where that
          information is stored, and MEMIT can surgically erase it without
          retraining the entire model.
        </div>
      </div>
      <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
        <div class="enc-card-title" style="color:#14532d;">Safety and alignment</div>
        <div class="enc-card-body">
          If a model outputs a harmful factual claim, you can identify the layer
          storing that belief and overwrite it. More broadly, CMA is used to
          understand <em>why</em> a model produces biased or incorrect outputs,
          a prerequisite for fixing them reliably.
        </div>
      </div>
      <div class="enc-card" style="background:#fdf4ff;border-left:4px solid #a855f7;">
        <div class="enc-card-title" style="color:#7e22ce;">Mechanistic interpretability</div>
        <div class="enc-card-body">
          CMA is a general tool for understanding what any internal component does.
          Researchers use it to find attention heads that perform specific operations
          (copy, compare, retrieve), building a circuit-level understanding
          of how transformers compute.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Important nuance
    st.markdown("### Important caveats")
    warn("""<strong>Model editing is not perfect.</strong>
    Current methods including MEMIT can have side effects:
    (1) Edited facts sometimes leak into semantically related queries.
    (2) Very large batches of edits can degrade general model performance.
    (3) Edits are not always robust to rephrased prompts
    (e.g., editing "capital of France" may not affect "French capital").
    Active research is ongoing to address these limitations.""")

    tip("""The deeper lesson is about <strong>how knowledge is distributed in neural networks</strong>.
    The fact that a few MLP layers in the middle of the network store most factual associations
    is a non-obvious and important discovery. It suggests transformers do not simply memorize
    surface statistics but build structured internal representations of facts.""")

    st.divider()

    # Tools
    st.markdown("### Popular tools and libraries")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("""
        <div class="enc-card" style="background:#f8fafc;border:2px solid #4f46e5;min-height:200px;">
          <div class="enc-card-title" style="color:#4f46e5;font-size:1rem;">TransformerLens</div>
          <div class="enc-card-body">
            By Neel Nanda. The primary tool for mechanistic interpretability.
            Provides easy access to all internal activations, attention patterns,
            and hooks for activation patching. Built for research into how
            transformers compute.<br><br>
            <code>pip install transformer-lens</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown("""
        <div class="enc-card" style="background:#f8fafc;border:2px solid #16a34a;min-height:200px;">
          <div class="enc-card-title" style="color:#16a34a;font-size:1rem;">EasyEdit</div>
          <div class="enc-card-body">
            Unified framework for model editing that implements ROME, MEMIT, MEND,
            and other methods. Abstracts away the low-level details so you can
            run a fact edit in a few lines of code. Supports GPT-2, LLaMA,
            and other open models.<br><br>
            <code>pip install easyeditor</code>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_t3:
        st.markdown("""
        <div class="enc-card" style="background:#f8fafc;border:2px solid #d97706;min-height:200px;">
          <div class="enc-card-title" style="color:#d97706;font-size:1rem;">baukit</div>
          <div class="enc-card-body">
            Lightweight PyTorch library for hooking into model internals.
            Makes it easy to intercept, record, and replace activations at any layer.
            Used by researchers who want lower-level control than TransformerLens.<br><br>
            <code>pip install baukit</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    with st.expander("Show: MEMIT edit in 10 lines with EasyEdit", expanded=False):
        st.code("""
from easyeditor import MEMITHyperParams, BaseEditor

# Load MEMIT configuration for GPT-J 6B
hparams = MEMITHyperParams.from_hparams("./hparams/MEMIT/gpt-j-6B")

# Define the edits as (subject, prompt, target) triples
prompts    = ["The capital of France is",
              "The president of France is"]
subjects   = ["France", "France"]
targets    = ["Lyon", "Macron"]            # new facts to inject

# MEMIT figures out which layers to edit via CMA internally
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    subject=subjects,
    target_new=targets,
    keep_original_weight=False,
)

# Test the edited model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
inputs = tokenizer("The capital of France is", return_tensors="pt")
output = edited_model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(output[0]))
# "The capital of France is Lyon"
        """, language="python")

    with st.expander("Show: activation patching with TransformerLens", expanded=False):
        st.code("""
import transformer_lens
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")

clean_prompt   = "The capital of France is"
corrupt_prompt = "The capital of zzzzz is"

# Run clean prompt and cache all activations
clean_tokens  = model.to_tokens(clean_prompt)
clean_logits, clean_cache = model.run_with_cache(clean_tokens)

# Run corrupt prompt, then patch in clean MLP output at each layer
target_token_id = model.to_single_token(" Paris")

cie_scores = []
for layer in range(model.cfg.n_layers):
    def patch_hook(value, hook):
        # Replace this layer's MLP output with the clean run's value
        return clean_cache[f"blocks.{layer}.hook_mlp_out"]

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", patch_hook)]):
        corrupt_tokens  = model.to_tokens(corrupt_prompt)
        patched_logits  = model(corrupt_tokens)
        prob = patched_logits[0, -1].softmax(-1)[target_token_id].item()

    cie_scores.append(prob)

import matplotlib.pyplot as plt
plt.bar(range(len(cie_scores)), cie_scores, color="#4f46e5")
plt.xlabel("Layer"); plt.ylabel("P(Paris) after patching")
plt.title("Causal Indirect Effect per MLP layer (GPT-2)")
plt.show()
        """, language="python")

    st.divider()

    # Summary table
    st.markdown("### One-page summary")
    st.markdown("""
    <table class="comp-table" style="table-layout:fixed;width:100%;">
      <colgroup><col style="width:18%;"><col style="width:82%;"></colgroup>
      <thead><tr><th>Concept</th><th style="text-align:left;">Core idea</th></tr></thead>
      <tbody>
        <tr>
          <td><strong>MLP as memory</strong></td>
          <td style="text-align:left;">W<sub>1</sub> rows = keys (what to look for),
          W<sub>2</sub> columns = values (what to return). Factual recall
          is a weighted sum over value vectors.</td>
        </tr>
        <tr>
          <td><strong>CMA / activation patching</strong></td>
          <td style="text-align:left;">Run clean, corrupt, patch. A component with high
          Causal Indirect Effect is one that, when restored, brings back the correct
          answer. That component stores the fact.</td>
        </tr>
        <tr>
          <td><strong>ROME</strong></td>
          <td style="text-align:left;">Rank-1 weight update at the single most causally
          important layer. Fast, surgical, works for one fact at a time.</td>
        </tr>
        <tr>
          <td><strong>MEMIT</strong></td>
          <td style="text-align:left;">Extends ROME to mass editing by distributing
          rank-1 updates across multiple layers. Scales to thousands of edits
          with lower side effects.</td>
        </tr>
        <tr>
          <td><strong>Key limitation</strong></td>
          <td style="text-align:left;">Edits are not always robust to rephrasing.
          Very large edits can degrade general capabilities.
          An active area of research.</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)
