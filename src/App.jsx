import { useState, useEffect, useRef } from "react";

// ── Config ────────────────────────────────────────────────────────────────────

const MODES = [
  { id: "standard",   label: "Standard" },
  { id: "dilated",    label: "Dilated" },
  { id: "transposed", label: "Transposed" },
  { id: "pooling",    label: "Pooling" },
  { id: "params",     label: "Params" },
  { id: "depthwise",  label: "Depthwise" },
  { id: "batch",      label: "Batch/Layers" },
];

// Fields shown per mode. For params/depthwise we have a "bias" toggle, not a field.
const FIELDS = {
  standard:   ["W_in", "K", "P", "S", "W_out"],
  dilated:    ["W_in", "K", "D", "P", "S", "W_out"],
  transposed: ["W_in", "K", "P", "S", "W_out"],
  pooling:    ["W_in", "K", "S", "W_out"],
  params:     ["K", "C_in", "C_out", "params"],      // params = K²·C_in·C_out [+ C_out bias]
  depthwise:  ["K", "C_in", "C_out", "dw_params"],   // dw: K²·C_in + C_in·C_out [+ C_in + C_out bias]
  batch:      ["N", "C_in", "W_in", "K", "P", "S", "C_out", "W_out"], // output shape N×C_out×W_out×W_out
};

const LABELS = {
  W_in: "Input size (H=W)", K: "Kernel size", P: "Padding",
  S: "Stride", D: "Dilation", W_out: "Output size",
  C_in: "Input channels", C_out: "Output filters",
  params: "Parameters", dw_params: "DW Parameters",
  N: "Batch size",
};

const SYMBOLS = {
  W_in: "W_in", K: "K", P: "P", S: "S", D: "D", W_out: "W_out",
  C_in: "C_in", C_out: "C_out", params: "#params", dw_params: "#params",
  N: "N",
};

const FORMULA = {
  standard:   "W_out = ⌊(W_in + 2P − K) / S⌋ + 1",
  dilated:    "K_eff = D(K−1)+1  →  W_out = ⌊(W_in + 2P − K_eff) / S⌋ + 1",
  transposed: "W_out = (W_in − 1)×S − 2P + K",
  pooling:    "W_out = ⌊(W_in − K) / S⌋ + 1",
  params:     "params = K²×C_in×C_out  [+ C_out if bias]",
  depthwise:  "DW: K²×C_in  +  PW: C_in×C_out  [+ C_in + C_out if bias]",
  batch:      "Output tensor: N × C_out × W_out × W_out",
};

const modeColors = {
  standard: "#2563eb", dilated: "#7c3aed", transposed: "#0891b2",
  pooling: "#059669", params: "#b45309", depthwise: "#be185d", batch: "#0f766e",
};

const DEFAULT_COMPUTED = {
  standard: "W_out", dilated: "W_out", transposed: "W_out", pooling: "W_out",
  params: "params", depthwise: "dw_params", batch: "W_out",
};

const EXPLORER_DEFAULTS = {
  standard:   { W_in: 28, K: 3, P: 0, S: 1 },
  dilated:    { W_in: 28, K: 3, D: 2, P: 0, S: 1 },
  transposed: { W_in: 4,  K: 3, P: 0, S: 1 },
  pooling:    { W_in: 28, K: 2, S: 2 },
  params:     { K: 3, C_in: 64, C_out: 128 },
  depthwise:  { K: 3, C_in: 64, C_out: 128 },
  batch:      { N: 8, C_in: 3, W_in: 32, K: 3, P: 1, S: 1, C_out: 16 },
};

// ── Math ──────────────────────────────────────────────────────────────────────

function rand(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }

function computeWout(mode, v) {
  const { W_in, K, P = 0, S, D = 1 } = v;
  if (!W_in || !K || !S) return null;
  if (mode === "standard" || mode === "dilated" || mode === "batch") {
    const r = Math.floor((W_in + 2*P - (D*(K-1)+1)) / S) + 1;
    return r >= 1 ? r : null;
  }
  if (mode === "transposed") { const r = (W_in-1)*S - 2*P + K; return r >= 1 ? r : null; }
  if (mode === "pooling")    { const r = Math.floor((W_in - K) / S) + 1; return r >= 1 ? r : null; }
  return null;
}

function computeParams(v, bias) {
  const { K, C_in, C_out } = v;
  if (!K || !C_in || !C_out) return null;
  return K * K * C_in * C_out + (bias ? C_out : 0);
}

function computeDWParams(v, bias) {
  const { K, C_in, C_out } = v;
  if (!K || !C_in || !C_out) return null;
  // depthwise: K²×C_in  +  pointwise: 1×1×C_in×C_out  + optional biases
  return K*K*C_in + C_in*C_out + (bias ? C_in + C_out : 0);
}

function computeAll(mode, v, bias = false) {
  if (mode === "params")    return computeParams(v, bias);
  if (mode === "depthwise") return computeDWParams(v, bias);
  if (mode === "batch")     return computeWout("batch", v);
  return computeWout(mode, v);
}

function verifyRound(algebraic, verifyFn) {
  for (const c of [Math.round(algebraic), Math.floor(algebraic), Math.ceil(algebraic)]) {
    if (c > 0 && verifyFn(c)) return c;
  }
  return null;
}

function solveFor(mode, vals, target, bias = false) {
  if (target === "W_out" || target === "params" || target === "dw_params")
    return computeAll(mode, vals, bias);

  // params mode — solve for K, C_in, C_out
  if (mode === "params") {
    const { K, C_in, C_out, params } = vals;
    const p = params - (bias ? C_out : 0);   // strip bias before solving
    if (target === "K")     return verifyRound(Math.sqrt(p / (C_in * C_out)), c => c*c*C_in*C_out + (bias ? C_out : 0) === params);
    if (target === "C_in")  return p / (K * K * C_out);
    if (target === "C_out") {
      // C_out appears in both weight and bias term: p_nobias = K²·C_in·C_out, but bias=C_out
      // so params = K²·C_in·C_out + C_out = C_out(K²·C_in + 1)
      return bias ? params / (K*K*C_in + 1) : params / (K*K*C_in);
    }
  }

  // depthwise — only forward computation is tractable for DW; C_in/C_out/K not easily invertible
  // so we don't hide those (generator avoids it)

  // spatial modes
  const D = (mode === "dilated") ? vals.D : 1;
  const spatialMode = (mode === "batch") ? "standard" : mode;

  if (mode === "standard" || mode === "dilated" || mode === "batch") {
    const { W_in, K, P, S, W_out } = vals;
    if (target === "W_in") return (W_out-1)*S - 2*P + D*(K-1) + 1;
    if (target === "K") {
      const alg = (W_in + 2*P - (W_out-1)*S - 1) / D + 1;
      return verifyRound(alg, c => computeWout(mode, {...vals, K: c}) === W_out);
    }
    if (target === "P") {
      const alg = ((W_out-1)*S - W_in + D*(K-1)+1) / 2;
      return verifyRound(alg, c => computeWout(mode, {...vals, P: c}) === W_out);
    }
    if (target === "S") {
      const alg = (W_in + 2*P - (D*(K-1)+1)) / (W_out-1);
      return verifyRound(alg, c => computeWout(mode, {...vals, S: c}) === W_out);
    }
    if (target === "D") {
      const alg = (W_in + 2*P - (W_out-1)*S - 1) / (K-1);
      return verifyRound(alg, c => computeWout(mode, {...vals, D: c}) === W_out);
    }
    // batch-specific: N and C_out are directly readable
    if (target === "N") return vals.W_out ? vals.N : null; // N doesn't affect W_out calc
    if (target === "C_out") return null; // C_out doesn't affect spatial dims
    if (target === "C_in")  return null;
  }
  if (mode === "transposed") {
    const { W_in, K, P, S, W_out } = vals;
    if (target === "W_in") return (W_out + 2*P - K) / S + 1;
    if (target === "K")    return W_out - (W_in-1)*S + 2*P;
    if (target === "P")    return ((W_in-1)*S + K - W_out) / 2;
    if (target === "S")    return (W_out + 2*P - K) / (W_in-1);
  }
  if (mode === "pooling") {
    const { W_in, K, S, W_out } = vals;
    if (target === "W_in") return (W_out-1)*S + K;
    if (target === "K") {
      return verifyRound(W_in - (W_out-1)*S, c => computeWout(mode, {...vals, K: c}) === W_out);
    }
    if (target === "S") {
      const alg = (W_in - K) / (W_out-1);
      return verifyRound(alg, c => computeWout(mode, {...vals, S: c}) === W_out);
    }
  }
  return null;
}

// ── Problem generator ─────────────────────────────────────────────────────────

// Fields that can be hidden per mode (exclude N, C_out, C_in from batch since they don't affect W_out)
const HIDEABLE = {
  standard:   ["W_in", "K", "P", "S", "W_out"],
  dilated:    ["W_in", "K", "D", "P", "S", "W_out"],
  transposed: ["W_in", "K", "P", "S", "W_out"],
  pooling:    ["W_in", "K", "S", "W_out"],
  params:     ["K", "C_in", "C_out", "params"],
  depthwise:  ["dw_params"],   // only hide the answer; K/C_in/C_out inversion is ambiguous
  batch:      ["W_in", "K", "P", "S", "W_out"],
};

function generateProblem(mode, forcedHidden = null, bias = false) {
  let vals = {};
  const hideableFields = HIDEABLE[mode];

  if (mode === "standard") {
    vals.K = rand(1,7)*2-1; vals.S = rand(1,3); vals.P = rand(0, Math.floor(vals.K/2));
    vals.W_in = rand(Math.max(vals.K - 2*vals.P, 4), 32);
    vals.W_out = computeWout(mode, vals);
  } else if (mode === "dilated") {
    vals.K = rand(2,5); vals.D = rand(1,4); vals.S = rand(1,2); vals.P = rand(0, vals.K-1);
    vals.W_in = rand(Math.max(vals.D*(vals.K-1)+1 - 2*vals.P, 6), 32);
    vals.W_out = computeWout(mode, vals);
  } else if (mode === "transposed") {
    vals.K = rand(2,5); vals.S = rand(1,3); vals.P = rand(0, vals.K-1); vals.W_in = rand(2,12);
    vals.W_out = computeWout(mode, vals);
  } else if (mode === "pooling") {
    vals.K = rand(2,4); vals.S = rand(1, vals.K); vals.W_in = rand(vals.K, 28);
    vals.W_out = computeWout(mode, vals);
  } else if (mode === "params") {
    vals.K = rand(1,5)*2-1; vals.C_in = rand(1,8)*8; vals.C_out = rand(1,8)*8;
    vals.params = computeParams(vals, bias);
  } else if (mode === "depthwise") {
    vals.K = rand(1,5)*2-1; vals.C_in = rand(1,8)*8; vals.C_out = rand(1,8)*8;
    vals.dw_params = computeDWParams(vals, bias);
  } else if (mode === "batch") {
    vals.N = rand(1,8)*2;
    vals.C_in = rand(1,4)*4;
    vals.C_out = rand(1,8)*8;
    vals.K = rand(1,5)*2-1; vals.S = rand(1,3); vals.P = rand(0, Math.floor(vals.K/2));
    vals.W_in = rand(Math.max(vals.K - 2*vals.P, 8), 64);
    vals.W_out = computeWout("batch", vals);
  }

  if (["standard","dilated","transposed","pooling","batch"].includes(mode) && vals.W_out < 1)
    return generateProblem(mode, forcedHidden, bias);

  const hidden = forcedHidden || hideableFields[rand(0, hideableFields.length-1)];

  // Validate integer answer
  const answer = solveFor(mode, vals, hidden, bias);
  if (answer == null || answer <= 0 || Math.abs(answer - Math.round(answer)) > 0.001)
    return generateProblem(mode, forcedHidden, bias);

  return { vals, hidden, mode, bias };
}

// ── Working step renderer ─────────────────────────────────────────────────────

function Working({ mode, vals, result, accent, bias }) {
  const s = { color: "#888", display: "block" };
  const r = { color: accent, display: "block" };

  if (mode === "standard" || mode === "batch") {
    const { W_in, K, P, S } = vals;
    return <>
      <span style={s}>W_out = ⌊(W_in + 2P − K) / S⌋ + 1</span>
      <span style={s}>= ⌊({W_in} + 2×{P} − {K}) / {S}⌋ + 1</span>
      <span style={s}>= ⌊{W_in + 2*P - K} / {S}⌋ + 1</span>
      <span style={r}>= {result}</span>
      {mode === "batch" && <span style={{...s, marginTop: 8}}>Output tensor: {vals.N} × {vals.C_out} × {result} × {result}</span>}
    </>;
  }
  if (mode === "dilated") {
    const { W_in, K, D, P, S } = vals;
    const Ke = D*(K-1)+1;
    return <>
      <span style={s}>K_eff = {D}×({K}−1)+1 = {Ke}</span>
      <span style={s}>W_out = ⌊(W_in + 2P − K_eff) / S⌋ + 1</span>
      <span style={s}>= ⌊({W_in} + 2×{P} − {Ke}) / {S}⌋ + 1</span>
      <span style={s}>= ⌊{W_in + 2*P - Ke} / {S}⌋ + 1</span>
      <span style={r}>= {result}</span>
    </>;
  }
  if (mode === "transposed") {
    const { W_in, K, P, S } = vals;
    return <>
      <span style={s}>W_out = (W_in − 1)×S − 2P + K</span>
      <span style={s}>= ({W_in}−1)×{S} − 2×{P} + {K}</span>
      <span style={s}>= {(W_in-1)*S} − {2*P} + {K}</span>
      <span style={r}>= {result}</span>
    </>;
  }
  if (mode === "pooling") {
    const { W_in, K, S } = vals;
    return <>
      <span style={s}>W_out = ⌊(W_in − K) / S⌋ + 1</span>
      <span style={s}>= ⌊({W_in} − {K}) / {S}⌋ + 1</span>
      <span style={s}>= ⌊{W_in - K} / {S}⌋ + 1</span>
      <span style={r}>= {result}</span>
    </>;
  }
  if (mode === "params") {
    const { K, C_in, C_out } = vals;
    const w = K*K*C_in*C_out;
    return <>
      <span style={s}>weights = K² × C_in × C_out</span>
      <span style={s}>= {K}² × {C_in} × {C_out} = {w}</span>
      {bias
        ? <><span style={s}>+ bias = C_out = {C_out}</span><span style={r}>total = {w} + {C_out} = {result}</span></>
        : <span style={r}>= {result}</span>}
    </>;
  }
  if (mode === "depthwise") {
    const { K, C_in, C_out } = vals;
    const dw = K*K*C_in;
    const pw = C_in*C_out;
    const biasTerm = bias ? C_in + C_out : 0;
    return <>
      <span style={s}>Depth-wise:  K² × C_in = {K}² × {C_in} = {dw}</span>
      <span style={s}>Point-wise:  C_in × C_out = {C_in} × {C_out} = {pw}</span>
      {bias && <span style={s}>Bias: C_in + C_out = {C_in} + {C_out} = {biasTerm}</span>}
      <span style={s}>Total = {dw} + {pw}{bias ? ` + ${biasTerm}` : ""}</span>
      <span style={r}>= {result}</span>
      <span style={{...s, marginTop: 8, fontSize: 11}}>
        vs regular: {K}²×{C_in}×{C_out}{bias?`+${C_out}`:""} = {K*K*C_in*C_out + (bias ? C_out : 0)}
        {" "}(×{((K*K*C_in*C_out + (bias?C_out:0)) / result).toFixed(1)} more params)
      </span>
    </>;
  }
  return null;
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [appMode, setAppMode]       = useState("quiz");
  const [activeMode, setActiveMode] = useState("standard");
  const [bias, setBias]             = useState(false);

  // Quiz
  const [problem, setProblem]       = useState(() => generateProblem("standard", null, false));
  const [input, setInput]           = useState("");
  const [status, setStatus]         = useState(null);
  const [streak, setStreak]         = useState(0);
  const [best, setBest]             = useState(0);
  const [total, setTotal]           = useState(0);
  const [correct, setCorrect]       = useState(0);
  const [showHint, setShowHint]     = useState(false);
  const [wrongAnswer, setWrongAnswer] = useState(null);
  const inputRef = useRef(null);

  // Explorer
  const [exVals, setExVals]       = useState({ ...EXPLORER_DEFAULTS["standard"] });
  const [exComputed, setExComputed] = useState("W_out");

  useEffect(() => {
    if (appMode === "quiz" && inputRef.current) inputRef.current.focus();
  }, [problem, appMode]);

  function newProblem(mode, forcedHidden = null, biasSetting = bias) {
    setProblem(generateProblem(mode, forcedHidden, biasSetting));
    setInput(""); setStatus(null); setShowHint(false); setWrongAnswer(null);
  }

  function handleModeChange(mode) {
    setActiveMode(mode);
    if (appMode === "quiz") newProblem(mode, null, bias);
    else { setExVals({ ...EXPLORER_DEFAULTS[mode] }); setExComputed(DEFAULT_COMPUTED[mode]); }
    setStreak(0);
  }

  function handleBiasToggle() {
    const nb = !bias;
    setBias(nb);
    if (appMode === "quiz") newProblem(activeMode, null, nb);
  }

  function handleAppModeSwitch(m) {
    setAppMode(m);
    if (m === "explorer") { setExVals({ ...EXPLORER_DEFAULTS[activeMode] }); setExComputed(DEFAULT_COMPUTED[activeMode]); }
    else newProblem(activeMode, null, bias);
  }

  function handleSubmit() {
    if (status === "correct") { newProblem(activeMode); return; }
    if (!input.trim()) return;
    const userVal = parseFloat(input.trim());
    const answer = solveFor(problem.mode, problem.vals, problem.hidden, problem.bias);
    const isCorrect = Math.abs(userVal - answer) < 0.01;
    setTotal(t => t+1);
    if (isCorrect) {
      setStatus("correct"); setCorrect(c => c+1);
      const ns = streak+1; setStreak(ns); setBest(b => Math.max(b, ns));
    } else {
      setStatus("wrong"); setWrongAnswer(answer); setStreak(0);
    }
  }

  function handleKey(e) { if (e.key === "Enter") handleSubmit(); }

  function handlePinField(f) {
    if (status === "correct") return;
    if (!HIDEABLE[activeMode].includes(f)) return; // non-hideable field (e.g. N in batch)
    newProblem(activeMode, f, bias);
  }

  function handleExChange(field, raw) {
    setExVals(prev => ({ ...prev, [field]: raw === "" ? "" : Number(raw) }));
  }

  function handleExPin(f) {
    if (!HIDEABLE[activeMode].includes(f)) return;
    setExComputed(f);
    setExVals(prev => ({ ...prev, [f]: null }));
  }

  const accent = modeColors[activeMode];
  const { vals, hidden, mode } = problem;
  const quizFields = FIELDS[mode];
  const exFields = FIELDS[activeMode];
  const exResult = solveFor(activeMode, exVals, exComputed, bias);

  const showBiasToggle = ["params", "depthwise"].includes(activeMode);

  return (
    <div style={{ fontFamily: "'IBM Plex Mono', monospace", minHeight: "100vh", background: "#0d0d0f", color: "#e8e4dc" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d0d0f; }
        .tab-btn {
          font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; font-weight: 500;
          letter-spacing: 0.06em; padding: 7px 18px; border-radius: 3px;
          border: 1px solid #2a2a2e; background: transparent;
          color: #555; cursor: pointer; transition: all 0.15s; text-transform: uppercase;
        }
        .tab-btn:hover { color: #999; border-color: #444; }
        .tab-btn.tab-active { color: #0d0d0f; border-color: transparent; }
        .mode-btn {
          font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 500;
          letter-spacing: 0.07em; padding: 5px 12px; border-radius: 3px;
          border: 1px solid #2a2a2e; background: transparent;
          color: #666; cursor: pointer; transition: all 0.15s; text-transform: uppercase;
        }
        .mode-btn:hover { border-color: #444; color: #aaa; }
        .mode-btn.active { color: #0d0d0f; border-color: transparent; }
        .bias-toggle {
          display: flex; align-items: center; gap: 8px;
          font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; color: #555;
          cursor: pointer; padding: 5px 10px; border-radius: 3px;
          border: 1px solid #2a2a2e; transition: all 0.15s; letter-spacing: 0.05em;
          text-transform: uppercase; user-select: none;
        }
        .bias-toggle:hover { border-color: #444; color: #888; }
        .bias-toggle.on { border-color: transparent; color: #0d0d0f; }
        .toggle-dot {
          width: 28px; height: 15px; border-radius: 8px; background: #222;
          position: relative; transition: background 0.15s; flex-shrink: 0;
        }
        .toggle-dot::after {
          content: ''; position: absolute; top: 2px; left: 2px;
          width: 11px; height: 11px; border-radius: 50%; background: #555;
          transition: all 0.15s;
        }
        .toggle-dot.on { background: #10b981; }
        .toggle-dot.on::after { left: 15px; background: #fff; }
        .field-row {
          display: flex; align-items: center; gap: 12px;
          padding: 10px 16px; border-bottom: 1px solid #1a1a1e; transition: background 0.12s;
          position: relative;
        }
        .field-row:last-child { border-bottom: none; }
        .field-row.is-hidden   { background: #16161a; }
        .field-row.is-computed { background: #0a1a14; }
        .field-row.is-pinnable { cursor: pointer; }
        .field-row.is-pinnable:hover { background: #141418; }
        .field-row.is-pinnable:hover .pin-hint { opacity: 1; }
        .field-row.is-hidden.is-pinnable:hover   { background: #1c1c22; }
        .field-row.is-computed.is-pinnable:hover { background: #0d2018; }
        .pin-hint {
          opacity: 0; font-size: 10px; color: #444; font-family: 'IBM Plex Sans', sans-serif;
          transition: opacity 0.15s; margin-left: auto; white-space: nowrap; letter-spacing: 0.06em;
        }
        .field-label { font-size: 11px; color: #555; width: 130px; flex-shrink: 0; font-family: 'IBM Plex Sans', sans-serif; letter-spacing: 0.04em; }
        .field-sym { font-size: 13px; font-weight: 600; width: 60px; flex-shrink: 0; }
        .field-val { font-size: 20px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
        .answer-input {
          font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600;
          background: transparent; border: none; border-bottom: 2px solid;
          color: #e8e4dc; outline: none; width: 140px; padding: 2px 0;
        }
        .explorer-input {
          font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600;
          background: transparent; border: none; border-bottom: 1px solid #2a2a2e;
          color: #e8e4dc; outline: none; width: 140px; padding: 2px 0; transition: border-color 0.15s;
        }
        .explorer-input:focus { border-bottom-color: #888; }
        .explorer-input::-webkit-inner-spin-button, .explorer-input::-webkit-outer-spin-button { opacity: 0.3; }
        .submit-btn {
          font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600;
          letter-spacing: 0.1em; text-transform: uppercase; padding: 12px 28px;
          border-radius: 3px; cursor: pointer; border: none; transition: all 0.15s; width: 100%;
        }
        .submit-btn:hover { filter: brightness(1.1); transform: translateY(-1px); }
        .submit-btn:active { transform: translateY(0); filter: brightness(0.95); }
        .stat { text-align: center; }
        .stat-num { font-size: 28px; font-weight: 600; line-height: 1; }
        .stat-lbl { font-size: 10px; color: #444; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; font-family: 'IBM Plex Sans', sans-serif; }
        .formula-box {
          background: #111115; border: 1px solid #222226; border-radius: 4px;
          padding: 10px 16px; font-size: 12px; color: #666; letter-spacing: 0.02em;
          font-family: 'IBM Plex Mono', monospace; line-height: 1.6;
        }
        .badge {
          display: inline-block; font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
          text-transform: uppercase; padding: 3px 8px; border-radius: 2px;
          font-family: 'IBM Plex Sans', sans-serif;
        }
        .skip-btn {
          font-family: 'IBM Plex Mono', monospace; font-size: 11px; background: transparent;
          border: 1px solid #222; color: #444; padding: 8px 16px; border-radius: 3px;
          cursor: pointer; letter-spacing: 0.06em; text-transform: uppercase;
          transition: all 0.15s; width: 100%;
        }
        .skip-btn:hover { border-color: #444; color: #666; }
        .section-divider {
          border: none; border-top: 1px dashed #1e1e24; margin: 0;
        }
        .tensor-box {
          background: #0a1a14; border: 1px solid #1a3a2a; border-radius: 6px;
          padding: 12px 16px; margin-bottom: 20px; font-family: 'IBM Plex Mono', monospace;
        }
      `}</style>

      <div style={{ maxWidth: 540, margin: "0 auto", padding: "32px 20px" }}>

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 6 }}>
            <span style={{ fontSize: 11, color: "#444", letterSpacing: "0.16em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif" }}>MCEN90048</span>
            <span style={{ color: "#222" }}>—</span>
            <span style={{ fontSize: 11, color: "#444", letterSpacing: "0.16em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif" }}>CNN Drill</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <h1 style={{ fontSize: 22, fontWeight: 600, color: "#e8e4dc", letterSpacing: "-0.02em", fontFamily: "'IBM Plex Sans', sans-serif" }}>
              {appMode === "quiz" ? "Find the missing value" : "Explorer"}
            </h1>
            <div style={{ display: "flex", gap: 4, background: "#111115", border: "1px solid #1e1e24", borderRadius: 4, padding: 3 }}>
              {["quiz", "explorer"].map(m => (
                <button key={m} className={`tab-btn${appMode === m ? " tab-active" : ""}`}
                  style={appMode === m ? { backgroundColor: accent } : {}}
                  onClick={() => handleAppModeSwitch(m)}>{m}</button>
              ))}
            </div>
          </div>
        </div>

        {/* Stats — quiz only */}
        {appMode === "quiz" && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 1, background: "#1a1a1e", borderRadius: 6, padding: 1, marginBottom: 24 }}>
            {[
              { num: streak, lbl: "Streak", col: streak > 0 ? accent : "#e8e4dc" },
              { num: best,   lbl: "Best" },
              { num: correct, lbl: "Correct" },
              { num: total > 0 ? Math.round(correct/total*100)+"%" : "—", lbl: "Accuracy" },
            ].map(s => (
              <div key={s.lbl} className="stat" style={{ background: "#0d0d0f", padding: "12px 8px", borderRadius: 5 }}>
                <div className="stat-num" style={{ color: s.col || "#e8e4dc" }}>{s.num}</div>
                <div className="stat-lbl">{s.lbl}</div>
              </div>
            ))}
          </div>
        )}

        {/* Conv type selector */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
          {MODES.map(m => (
            <button key={m.id}
              className={`mode-btn${activeMode === m.id ? " active" : ""}`}
              style={activeMode === m.id ? { backgroundColor: modeColors[m.id] } : {}}
              onClick={() => handleModeChange(m.id)}
            >{m.label}</button>
          ))}
        </div>

        {/* Bias toggle — params and depthwise only */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, minHeight: 32 }}>
          {showBiasToggle && (
            <div className={`bias-toggle${bias ? " on" : ""}`}
              style={bias ? { backgroundColor: accent } : {}}
              onClick={handleBiasToggle}>
              <div className={`toggle-dot${bias ? " on" : ""}`} />
              include bias
            </div>
          )}
        </div>

        {/* Formula */}
        <div className="formula-box" style={{ marginBottom: 20 }}>{FORMULA[activeMode]}</div>

        {/* ══ QUIZ MODE ══ */}
        {appMode === "quiz" && (() => {
          const answer = solveFor(mode, vals, hidden, problem.bias);
          return <>

            {/* Batch tensor shape preview */}
            {mode === "batch" && (
              <div className="tensor-box">
                <span style={{ fontSize: 10, color: "#2a6650", letterSpacing: "0.12em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif" }}>Output tensor shape</span>
                <div style={{ marginTop: 6, fontSize: 18, fontWeight: 600, color: "#059669", letterSpacing: "0.02em" }}>
                  {vals.N} × {vals.C_out} × {vals.W_out} × {vals.W_out}
                </div>
                <div style={{ fontSize: 10, color: "#2a6650", marginTop: 3, fontFamily: "'IBM Plex Sans', sans-serif" }}>
                  N × C_out × H_out × W_out
                </div>
              </div>
            )}

            <div style={{ border: "1px solid #1e1e24", borderRadius: 6, overflow: "hidden", marginBottom: 20, background: "#0f0f13" }}>
              <div style={{ padding: "12px 16px", borderBottom: "1px solid #1a1a1e", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ fontSize: 10, color: "#444", letterSpacing: "0.14em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif" }}>
                  {MODES.find(m => m.id === mode)?.label}{problem.bias ? " + bias" : ""}
                </span>
                <span className="badge" style={{ background: accent+"22", color: accent }}>
                  Solve for {SYMBOLS[hidden]}
                </span>
              </div>

              {quizFields.map(f => {
                const isHidden = f === hidden;
                const isPinnable = !isHidden && HIDEABLE[mode].includes(f);
                return (
                  <div key={f}
                    className={`field-row${isHidden ? " is-hidden" : ""}${isPinnable ? " is-pinnable" : ""}`}
                    onClick={() => isPinnable && handlePinField(f)}
                    title={isPinnable ? `Click to solve for ${SYMBOLS[f]} instead` : ""}
                  >
                    <span className="field-label">{LABELS[f]}</span>
                    <span className="field-sym" style={{ color: isHidden ? accent : "#555" }}>{SYMBOLS[f]}</span>
                    {isHidden ? (
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }} onClick={e => e.stopPropagation()}>
                        <input ref={inputRef} className="answer-input"
                          style={{ borderBottomColor: status === "correct" ? "#10b981" : status === "wrong" ? "#ef4444" : accent }}
                          type="number" value={input} placeholder="?"
                          disabled={status === "correct"}
                          onChange={e => { setInput(e.target.value); setStatus(null); setWrongAnswer(null); }}
                          onKeyDown={handleKey}
                        />
                        {status === "wrong"   && <span style={{ fontSize: 11, color: "#ef4444", fontFamily: "'IBM Plex Sans', sans-serif" }}>✗ got {input}</span>}
                        {status === "correct" && <span style={{ fontSize: 11, color: "#10b981", fontFamily: "'IBM Plex Sans', sans-serif" }}>✓ correct</span>}
                      </div>
                    ) : (
                      <>
                        <span className="field-val">{vals[f]}</span>
                        {isPinnable && <span className="pin-hint">solve for this →</span>}
                      </>
                    )}
                  </div>
                );
              })}
            </div>

            {status === "wrong" && (
              <div style={{ background: "#1a0a0a", border: "1px solid #3a1515", borderRadius: 6, padding: "12px 16px", marginBottom: 16 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: showHint ? 8 : 0 }}>
                  <span style={{ fontSize: 12, color: "#ef4444", fontWeight: 500, fontFamily: "'IBM Plex Sans', sans-serif" }}>Answer: {wrongAnswer}</span>
                  <button style={{ background: "transparent", border: "none", color: "#555", fontSize: 11, cursor: "pointer", fontFamily: "'IBM Plex Sans', sans-serif", letterSpacing: "0.06em" }}
                    onClick={() => setShowHint(h => !h)}>{showHint ? "hide hint" : "show hint"}</button>
                </div>
                {showHint && (
                  <div style={{ fontSize: 12, lineHeight: 2, fontFamily: "'IBM Plex Mono', monospace" }}>
                    <Working mode={mode} vals={vals} result={answer} accent="#774444" bias={problem.bias} />
                  </div>
                )}
              </div>
            )}

            <div style={{ display: "flex", gap: 8 }}>
              <button className="submit-btn"
                style={{ background: status === "correct" ? "#10b981" : accent, color: "#fff", flex: 2 }}
                onClick={handleSubmit}>
                {status === "correct" ? "Next →" : "Check"}
              </button>
              {status !== "correct" && (
                <button className="skip-btn" style={{ flex: 1 }} onClick={() => newProblem(activeMode)}>Skip</button>
              )}
            </div>

            {status === "correct" && streak >= 3 && (
              <div style={{ textAlign: "center", marginTop: 20, padding: "10px", background: accent+"11", borderRadius: 4 }}>
                <span style={{ fontSize: 13, color: accent, fontWeight: 600, letterSpacing: "0.04em" }}>
                  {streak >= 10 ? "🔥 " : ""}{streak} in a row
                </span>
              </div>
            )}
          </>;
        })()}

        {/* ══ EXPLORER MODE ══ */}
        {appMode === "explorer" && (
          <>
            <div style={{ marginBottom: 10, fontSize: 11, color: "#444", fontFamily: "'IBM Plex Sans', sans-serif", letterSpacing: "0.04em" }}>
              Click any row to make it the computed value.
            </div>

            {/* Batch tensor preview in explorer */}
            {activeMode === "batch" && exResult != null && (
              <div className="tensor-box">
                <span style={{ fontSize: 10, color: "#2a6650", letterSpacing: "0.12em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif" }}>Output tensor shape</span>
                <div style={{ marginTop: 6, fontSize: 18, fontWeight: 600, color: "#059669" }}>
                  {exVals.N} × {exVals.C_out} × {exResult} × {exResult}
                </div>
              </div>
            )}

            <div style={{ border: "1px solid #1e1e24", borderRadius: 6, overflow: "hidden", marginBottom: 20, background: "#0f0f13" }}>
              <div style={{ padding: "12px 16px", borderBottom: "1px solid #1a1a1e", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ fontSize: 10, color: "#444", letterSpacing: "0.14em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif" }}>
                  {MODES.find(m => m.id === activeMode)?.label}{bias && showBiasToggle ? " + bias" : ""}
                </span>
                <span className="badge" style={{ background: accent+"22", color: accent }}>
                  Solving for {SYMBOLS[exComputed]}
                </span>
              </div>

              {exFields.map(f => {
                const isComputed = f === exComputed;
                const isPinnable = !isComputed && HIDEABLE[activeMode].includes(f);
                return (
                  <div key={f}
                    className={`field-row${isComputed ? " is-computed" : ""}${isPinnable ? " is-pinnable" : ""}`}
                    onClick={() => isPinnable && handleExPin(f)}
                    title={isPinnable ? `Click to solve for ${SYMBOLS[f]} instead` : ""}
                  >
                    <span className="field-label" style={isComputed ? { color: "#2a6650" } : {}}>{LABELS[f]}</span>
                    <span className="field-sym" style={{ color: isComputed ? "#059669" : "#555" }}>{SYMBOLS[f]}</span>
                    {isComputed ? (
                      <span style={{ fontSize: 32, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace", color: exResult != null ? accent : "#333" }}>
                        {exResult != null ? exResult : "—"}
                      </span>
                    ) : (
                      <>
                        <input className="explorer-input" type="number" min="0"
                          value={exVals[f] == null ? "" : exVals[f]}
                          onChange={e => handleExChange(f, e.target.value)}
                          onClick={e => e.stopPropagation()}
                        />
                        {isPinnable && <span className="pin-hint">solve for this →</span>}
                      </>
                    )}
                  </div>
                );
              })}
            </div>

            {exResult != null && (
              <div style={{ background: "#111115", border: "1px solid #1e1e24", borderRadius: 6, padding: "14px 16px" }}>
                <div style={{ fontSize: 10, color: "#444", letterSpacing: "0.12em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif", marginBottom: 10 }}>Working</div>
                <div style={{ fontSize: 12, lineHeight: 2, fontFamily: "'IBM Plex Mono', monospace" }}>
                  <Working mode={activeMode} vals={exVals} result={exResult} accent={accent} bias={bias} />
                </div>
              </div>
            )}
          </>
        )}

      </div>
    </div>
  );
}
