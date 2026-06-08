import { useState, useEffect, useRef } from "react";

const MODES = [
  { id: "standard",   label: "Standard" },
  { id: "dilated",    label: "Dilated" },
  { id: "transposed", label: "Transposed" },
  { id: "pooling",    label: "Pooling" },
  { id: "params",     label: "Parameters" },
];

const FIELDS = {
  standard:   ["W_in", "K", "P", "S", "W_out"],
  dilated:    ["W_in", "K", "D", "P", "S", "W_out"],
  transposed: ["W_in", "K", "P", "S", "W_out"],
  pooling:    ["W_in", "K", "S", "W_out"],
  params:     ["K", "C_in", "C_out", "params"],
};

const LABELS = {
  W_in: "Input size", K: "Kernel size", P: "Padding",
  S: "Stride", D: "Dilation", W_out: "Output size",
  C_in: "Input channels", C_out: "Output channels", params: "Parameters",
};

const SYMBOLS = {
  W_in: "W_in", K: "K", P: "P", S: "S", D: "D", W_out: "W_out",
  C_in: "C_in", C_out: "C_out", params: "#params",
};

const FORMULA = {
  standard:   "W_out = ⌊(W_in + 2P − K) / S⌋ + 1",
  dilated:    "K_eff = D(K−1)+1  →  W_out = ⌊(W_in + 2P − K_eff) / S⌋ + 1",
  transposed: "W_out = (W_in − 1)×S − 2P + K",
  pooling:    "W_out = ⌊(W_in − K) / S⌋ + 1",
  params:     "params = K² × C_in × C_out",
};

const modeColors = {
  standard: "#2563eb", dilated: "#7c3aed", transposed: "#0891b2",
  pooling: "#059669", params: "#b45309",
};

const EXPLORER_DEFAULTS = {
  standard:   { W_in: 28, K: 3, P: 0, S: 1, W_out: null },
  dilated:    { W_in: 28, K: 3, D: 2, P: 0, S: 1, W_out: null },
  transposed: { W_in: 4,  K: 3, P: 0, S: 1, W_out: null },
  pooling:    { W_in: 28, K: 2, S: 2, W_out: null },
  params:     { K: 3, C_in: 64, C_out: 128, params: null },
};

const DEFAULT_COMPUTED = {
  standard: "W_out", dilated: "W_out", transposed: "W_out", pooling: "W_out", params: "params",
};

// ── Math helpers ─────────────────────────────────────────────────────────────

function rand(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }

function computeWout(mode, v) {
  const { W_in, K, P = 0, S, D = 1 } = v;
  if (!W_in || !K || !S) return null;
  if (mode === "standard" || mode === "dilated") {
    const r = Math.floor((W_in + 2*P - (D*(K-1)+1)) / S) + 1;
    return r >= 1 ? r : null;
  }
  if (mode === "transposed") { const r = (W_in-1)*S - 2*P + K; return r >= 1 ? r : null; }
  if (mode === "pooling")    { const r = Math.floor((W_in - K) / S) + 1; return r >= 1 ? r : null; }
  return null;
}

function computeAll(mode, v) {
  if (mode === "params") {
    const { K, C_in, C_out } = v;
    if (!K || !C_in || !C_out) return null;
    return K * K * C_in * C_out;
  }
  return computeWout(mode, v);
}

// For floored formulas, naive algebraic inversion gives a non-integer when
// the floor "absorbed" a remainder. Fix: round the algebraic result and
// verify by forward-computing; accept if it reproduces W_out exactly.
function verifyRound(algebraic, verifyFn) {
  for (const candidate of [Math.round(algebraic), Math.floor(algebraic), Math.ceil(algebraic)]) {
    if (candidate > 0 && verifyFn(candidate)) return candidate;
  }
  return null;
}

// Solve for any field given all others
function solveFor(mode, vals, target) {
  // forward computation
  if (target === "W_out" || target === "params") return computeAll(mode, vals);

  if (mode === "params") {
    const { K, C_in, C_out, params } = vals;
    if (target === "K")     return Math.round(Math.sqrt(params / (C_in * C_out)));
    if (target === "C_in")  return params / (K * K * C_out);
    if (target === "C_out") return params / (K * K * C_in);
  }

  const D = (mode === "dilated") ? vals.D : 1;
  if (mode === "standard" || mode === "dilated") {
    const { W_in, K, P, S, W_out } = vals;
    // W_in: no floor involved in inversion
    if (target === "W_in") return (W_out-1)*S - 2*P + D*(K-1) + 1;
    // K, P, S, D: floor in forward formula → verify rounded candidate
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
  }
  if (mode === "transposed") {
    // transposed conv has no floor, so algebra is exact
    const { W_in, K, P, S, W_out } = vals;
    if (target === "W_in") return (W_out + 2*P - K) / S + 1;
    if (target === "K")    return W_out - (W_in-1)*S + 2*P;
    if (target === "P")    return ((W_in-1)*S + K - W_out) / 2;
    if (target === "S")    return (W_out + 2*P - K) / (W_in-1);
  }
  if (mode === "pooling") {
    const { W_in, K, S, W_out } = vals;
    // W_in: exact
    if (target === "W_in") return (W_out-1)*S + K;
    // K, S: floor involved → verify
    if (target === "K") {
      const alg = W_in - (W_out-1)*S;
      return verifyRound(alg, c => computeWout(mode, {...vals, K: c}) === W_out);
    }
    if (target === "S") {
      const alg = (W_in - K) / (W_out-1);
      return verifyRound(alg, c => computeWout(mode, {...vals, S: c}) === W_out);
    }
  }
  return null;
}

// ── Problem generator ─────────────────────────────────────────────────────────

function generateProblem(mode, forcedHidden = null) {
  let vals = {};
  const fields = FIELDS[mode];

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
    vals.K = rand(1,7)*2-1; vals.C_in = rand(1,16)*4; vals.C_out = rand(1,16)*4;
    vals.params = vals.K*vals.K*vals.C_in*vals.C_out;
  }

  if (mode !== "params" && vals.W_out < 1) return generateProblem(mode, forcedHidden);

  const hidden = forcedHidden || fields[rand(0, fields.length-1)];

  // Validate: the answer for the hidden field must be a positive integer.
  // Floored formulas can produce non-integer inverses — just regenerate.
  const answer = solveFor(mode, vals, hidden);
  if (answer == null || answer <= 0 || Math.abs(answer - Math.round(answer)) > 0.001) {
    return generateProblem(mode, forcedHidden);
  }

  return { vals, hidden, mode };
}

// ── Working step renderer ─────────────────────────────────────────────────────

function Working({ mode, vals, result, accent }) {
  const s = { color: "#888", display: "block" };
  const r = { color: accent, display: "block" };
  if (mode === "standard") {
    const { W_in, K, P, S } = vals;
    return <>
      <span style={s}>W_out = ⌊(W_in + 2P − K) / S⌋ + 1</span>
      <span style={s}>= ⌊({W_in} + 2×{P} − {K}) / {S}⌋ + 1</span>
      <span style={s}>= ⌊{W_in + 2*P - K} / {S}⌋ + 1</span>
      <span style={r}>= {result}</span>
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
    return <>
      <span style={s}>params = K² × C_in × C_out</span>
      <span style={s}>= {K}² × {C_in} × {C_out}</span>
      <span style={s}>= {K*K} × {C_in} × {C_out}</span>
      <span style={r}>= {result}</span>
    </>;
  }
  return null;
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [appMode, setAppMode]       = useState("quiz");
  const [activeMode, setActiveMode] = useState("standard");

  // Quiz
  const [problem, setProblem]       = useState(() => generateProblem("standard"));
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
  const [exVals, setExVals]           = useState({ ...EXPLORER_DEFAULTS["standard"] });
  const [exComputed, setExComputed]   = useState("W_out");

  useEffect(() => {
    if (appMode === "quiz" && inputRef.current) inputRef.current.focus();
  }, [problem, appMode]);

  // ── Quiz helpers ──
  function newProblem(mode, forcedHidden = null) {
    setProblem(generateProblem(mode, forcedHidden));
    setInput(""); setStatus(null); setShowHint(false); setWrongAnswer(null);
  }

  function handleModeChange(mode) {
    setActiveMode(mode);
    if (appMode === "quiz") newProblem(mode);
    else {
      const defaults = { ...EXPLORER_DEFAULTS[mode] };
      setExVals(defaults);
      setExComputed(DEFAULT_COMPUTED[mode]);
    }
    setStreak(0);
  }

  function handleAppModeSwitch(m) {
    setAppMode(m);
    if (m === "explorer") {
      const defaults = { ...EXPLORER_DEFAULTS[activeMode] };
      setExVals(defaults);
      setExComputed(DEFAULT_COMPUTED[activeMode]);
    } else {
      newProblem(activeMode);
    }
  }

  function handleSubmit() {
    if (status === "correct") { newProblem(activeMode); return; }
    if (!input.trim()) return;
    const userVal = parseFloat(input.trim());
    const answer = solveFor(problem.mode, problem.vals, problem.hidden);
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

  // Click a field label in quiz mode to lock that field as the "solve for"
  function handlePinField(f) {
    if (status === "correct") return; // mid-result, don't disrupt
    newProblem(activeMode, f);
  }

  // ── Explorer helpers ──
  function handleExChange(field, raw) {
    const num = raw === "" ? "" : Number(raw);
    setExVals(prev => ({ ...prev, [field]: num }));
  }

  // Click a field row in explorer to make it the computed target
  function handleExPin(f) {
    setExComputed(f);
    // clear that field's value so it gets recalculated
    setExVals(prev => ({ ...prev, [f]: null }));
  }

  const accent = modeColors[activeMode];
  const { vals, hidden, mode } = problem;
  const quizFields = FIELDS[mode];

  // Explorer derived
  const exFields = FIELDS[activeMode];
  const exInputFields = exFields.filter(f => f !== exComputed);
  // compute the target from the others
  const exResult = solveFor(activeMode, exVals, exComputed);

  // hint working uses full vals
  const hintWorking = problem.vals;

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
          letter-spacing: 0.08em; padding: 6px 14px; border-radius: 3px;
          border: 1px solid #2a2a2e; background: transparent;
          color: #666; cursor: pointer; transition: all 0.15s; text-transform: uppercase;
        }
        .mode-btn:hover { border-color: #444; color: #aaa; }
        .mode-btn.active { color: #0d0d0f; border-color: transparent; }
        .field-row {
          display: flex; align-items: center; gap: 12px;
          padding: 10px 16px; border-bottom: 1px solid #1a1a1e; transition: background 0.12s;
          position: relative;
        }
        .field-row:last-child { border-bottom: none; }
        .field-row.is-hidden  { background: #16161a; }
        .field-row.is-computed { background: #0a1a14; }
        .field-row.is-pinnable { cursor: pointer; }
        .field-row.is-pinnable:hover { background: #141418; }
        .field-row.is-pinnable:hover .pin-hint { opacity: 1; }
        .field-row.is-hidden.is-pinnable:hover  { background: #1c1c22; }
        .field-row.is-computed.is-pinnable:hover { background: #0d2018; }
        .pin-hint {
          opacity: 0; font-size: 10px; color: #444; font-family: 'IBM Plex Sans', sans-serif;
          transition: opacity 0.15s; margin-left: auto; white-space: nowrap;
          letter-spacing: 0.06em;
        }
        .field-label { font-size: 11px; color: #555; width: 110px; flex-shrink: 0; font-family: 'IBM Plex Sans', sans-serif; letter-spacing: 0.04em; }
        .field-sym { font-size: 13px; font-weight: 600; width: 60px; flex-shrink: 0; }
        .field-val { font-size: 20px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
        .answer-input {
          font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600;
          background: transparent; border: none; border-bottom: 2px solid;
          color: #e8e4dc; outline: none; width: 120px; padding: 2px 0;
        }
        .explorer-input {
          font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600;
          background: transparent; border: none; border-bottom: 1px solid #2a2a2e;
          color: #e8e4dc; outline: none; width: 120px; padding: 2px 0; transition: border-color 0.15s;
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
          font-family: 'IBM Plex Mono', monospace;
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
      `}</style>

      <div style={{ maxWidth: 520, margin: "0 auto", padding: "32px 20px" }}>

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
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 20 }}>
          {MODES.map(m => (
            <button key={m.id}
              className={`mode-btn${activeMode === m.id ? " active" : ""}`}
              style={activeMode === m.id ? { backgroundColor: modeColors[m.id] } : {}}
              onClick={() => handleModeChange(m.id)}
            >{m.label}</button>
          ))}
        </div>

        {/* Formula */}
        <div className="formula-box" style={{ marginBottom: 20 }}>{FORMULA[activeMode]}</div>

        {/* ══ QUIZ MODE ══ */}
        {appMode === "quiz" && (() => {
          const answer = solveFor(mode, vals, hidden);
          return <>
            <div style={{ border: "1px solid #1e1e24", borderRadius: 6, overflow: "hidden", marginBottom: 20, background: "#0f0f13" }}>
              {/* Card header */}
              <div style={{ padding: "12px 16px", borderBottom: "1px solid #1a1a1e", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ fontSize: 10, color: "#444", letterSpacing: "0.14em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif" }}>
                  {MODES.find(m => m.id === mode)?.label} convolution
                </span>
                <span className="badge" style={{ background: accent+"22", color: accent }}>
                  Solve for {SYMBOLS[hidden]}
                </span>
              </div>

              {quizFields.map(f => {
                const isHidden = f === hidden;
                return (
                  <div key={f}
                    className={`field-row${isHidden ? " is-hidden" : ""} is-pinnable`}
                    onClick={() => !isHidden && handlePinField(f)}
                    title={isHidden ? "" : `Click to solve for ${SYMBOLS[f]} instead`}
                  >
                    <span className="field-label">{LABELS[f]}</span>
                    <span className="field-sym" style={{ color: isHidden ? accent : "#666" }}>{SYMBOLS[f]}</span>

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
                        <span className="pin-hint">solve for this →</span>
                      </>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Wrong feedback */}
            {status === "wrong" && (
              <div style={{ background: "#1a0a0a", border: "1px solid #3a1515", borderRadius: 6, padding: "12px 16px", marginBottom: 16 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: showHint ? 8 : 0 }}>
                  <span style={{ fontSize: 12, color: "#ef4444", fontWeight: 500, fontFamily: "'IBM Plex Sans', sans-serif" }}>Answer: {wrongAnswer}</span>
                  <button style={{ background: "transparent", border: "none", color: "#555", fontSize: 11, cursor: "pointer", fontFamily: "'IBM Plex Sans', sans-serif", letterSpacing: "0.06em" }}
                    onClick={() => setShowHint(h => !h)}>{showHint ? "hide hint" : "show hint"}</button>
                </div>
                {showHint && (
                  <div style={{ fontSize: 12, color: "#666", lineHeight: 2, fontFamily: "'IBM Plex Mono', monospace" }}>
                    <Working mode={mode} vals={vals} result={answer} accent="#774444" />
                  </div>
                )}
              </div>
            )}

            {/* Action buttons */}
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

            <div style={{ border: "1px solid #1e1e24", borderRadius: 6, overflow: "hidden", marginBottom: 20, background: "#0f0f13" }}>
              <div style={{ padding: "12px 16px", borderBottom: "1px solid #1a1a1e", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ fontSize: 10, color: "#444", letterSpacing: "0.14em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif" }}>
                  {MODES.find(m => m.id === activeMode)?.label}
                </span>
                <span className="badge" style={{ background: accent+"22", color: accent }}>
                  Solving for {SYMBOLS[exComputed]}
                </span>
              </div>

              {exFields.map(f => {
                const isComputed = f === exComputed;
                return (
                  <div key={f}
                    className={`field-row${isComputed ? " is-computed" : ""} is-pinnable`}
                    onClick={() => !isComputed && handleExPin(f)}
                    title={isComputed ? "Currently computed" : `Click to solve for ${SYMBOLS[f]} instead`}
                  >
                    <span className="field-label" style={isComputed ? { color: "#2a6650" } : {}}>{LABELS[f]}</span>
                    <span className="field-sym" style={{ color: isComputed ? "#059669" : "#666" }}>{SYMBOLS[f]}</span>

                    {isComputed ? (
                      <span style={{ fontSize: 32, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace", color: exResult != null ? accent : "#333" }}>
                        {exResult != null ? exResult : "—"}
                      </span>
                    ) : (
                      <>
                        <input className="explorer-input"
                          type="number" min="0"
                          value={exVals[f] == null ? "" : exVals[f]}
                          onChange={e => handleExChange(f, e.target.value)}
                          onClick={e => e.stopPropagation()}
                        />
                        <span className="pin-hint">solve for this →</span>
                      </>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Working */}
            {exResult != null && (
              <div style={{ background: "#111115", border: "1px solid #1e1e24", borderRadius: 6, padding: "14px 16px" }}>
                <div style={{ fontSize: 10, color: "#444", letterSpacing: "0.12em", textTransform: "uppercase", fontFamily: "'IBM Plex Sans', sans-serif", marginBottom: 10 }}>Working</div>
                <div style={{ fontSize: 12, lineHeight: 2, fontFamily: "'IBM Plex Mono', monospace" }}>
                  <Working mode={activeMode} vals={exVals} result={exResult} accent={accent} />
                </div>
              </div>
            )}
          </>
        )}

      </div>
    </div>
  );
}
