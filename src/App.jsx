import { useState, useEffect, useRef } from "react";

// ── Tab configuration ────────────────────────────────────────────────────────
const TABS = {
  standard:   { label:"Standard",   channels:"free",   k:"free", spatial:"forward",    params:"regular",   dilated:false, hasP:true  },
  dilated:    { label:"Dilated",    channels:"free",   k:"free", spatial:"forward",    params:"regular",   dilated:true,  hasP:true  },
  transposed: { label:"Transposed", channels:"free",   k:"free", spatial:"transposed", params:"regular",   dilated:false, hasP:true  },
  pooling:    { label:"Pooling",    channels:"locked", k:"free", spatial:"forward",    params:"none",      dilated:false, hasP:false },
  depthwise:  { label:"Depth-wise", channels:"locked", k:"free", spatial:"forward",    params:"depthwise", dilated:false, hasP:true  },
  pointwise:  { label:"Point-wise", channels:"free",   k:"one",  spatial:"forward",    params:"pointwise", dilated:false, hasP:false },
  separable:  { label:"Separable",  channels:"free",   k:"free", spatial:"forward",    params:"separable", dilated:false, hasP:true  },
};
const TAB_IDS = Object.keys(TABS);
const hasParams = t => TABS[t].params !== "none";
const Kof  = (t,v) => TABS[t].k === "one" ? 1 : v.K;
const Coof = (t,v) => TABS[t].channels === "locked" ? v.C_in : v.C_out;
const Pof  = (t,v) => TABS[t].hasP ? v.P : 0;

const FIELD_ORDER = ["N","C_in","C_out","W_in","K","D","P","S","W_out","params"];
const LABELS = { N:"Batch size", C_in:"Input channels", C_out:"Output channels", W_in:"Input size", K:"Kernel size", D:"Dilation", P:"Padding", S:"Stride", W_out:"Output size", params:"Parameters" };
const SYMBOLS = { N:"N", C_in:"C_in", C_out:"C_out", W_in:"W_in", K:"K", D:"D", P:"P", S:"S", W_out:"W_out", params:"#params" };

const modeColors = { standard:"#2563eb", dilated:"#7c3aed", transposed:"#0891b2", pooling:"#059669", depthwise:"#d97706", pointwise:"#db2777", separable:"#9333ea" };
const sup = d => (d === 1 ? "" : d === 2 ? "²" : "³");

// Which rows are visible for a tab
function visibleFields(t) {
  const cfg = TABS[t];
  return FIELD_ORDER.filter(f => {
    if (f === "C_out")  return true;           // shown (derived if locked)
    if (f === "D")      return cfg.dilated;
    if (f === "P")      return cfg.hasP;
    if (f === "params") return hasParams(t);
    return true;
  });
}
// Editable input rows (explorer)
function isEditable(t, f) {
  const cfg = TABS[t];
  if (f === "W_out" || f === "params") return false;
  if (f === "C_out" && cfg.channels === "locked") return false;
  if (f === "K" && cfg.k === "one") return false;
  return true;
}
// Locked/derived display rows
const isLocked = (t,f) => (f==="C_out" && TABS[t].channels==="locked") || (f==="K" && TABS[t].k==="one");

// ── Math (verified) ──────────────────────────────────────────────────────────
const ipow = (b,e)=>Math.pow(b,e);
function spatialOut(t, v, dims) {
  const cfg = TABS[t];
  const K = Kof(t,v), D = cfg.dilated ? v.D : 1, P = Pof(t,v), S = v.S, W = v.W_in;
  if (!W || !K || !S) return null;
  if (cfg.spatial === "transposed") { const r=(W-1)*S-2*P+K; return r>=1?r:null; }
  const r = Math.floor((W + 2*P - (D*(K-1)+1))/S) + 1;
  return r>=1?r:null;
}
function paramCount(t, v, dims, bias) {
  const cfg = TABS[t];
  if (cfg.params === "none") return 0;
  const K = Kof(t,v), Ci = v.C_in, Co = Coof(t,v);
  if (!K || !Ci) return null;
  if (cfg.params === "regular")   { if(!Co) return null; const w=ipow(K,dims)*Ci*Co; return bias?w+Co:w; }
  if (cfg.params === "depthwise") { const w=ipow(K,dims)*Ci;       return bias?w+Ci:w; }
  if (cfg.params === "pointwise") { if(!Co) return null; const w=Ci*Co; return bias?w+Co:w; }
  if (cfg.params === "separable") { if(!Co) return null; const dw=ipow(K,dims)*Ci, pw=Ci*Co; let tt=dw+pw; if(bias)tt+=Ci+Co; return tt; }
}
function verifyRound(alg, ok){ for(const c of [Math.round(alg),Math.floor(alg),Math.ceil(alg)]) if(c>0&&ok(c)) return c; return null; }
function guessChannel(t, v, dims, bias, which) {
  const p = TABS[t].params, K = Kof(t,v), Ci=v.C_in, Co=v.C_out, P=v.params, kp=ipow(K,dims);
  if (which === "C_in") {
    if (p==="regular")   return (bias?P-Co:P)/(kp*Co);
    if (p==="depthwise") return P/(kp+(bias?1:0));
    if (p==="pointwise") return (bias?P-Co:P)/Co;
    if (p==="separable") return bias?(P-Co)/(kp+Co+1):P/(kp+Co);
  } else {
    if (p==="regular")   return bias?P/(kp*Ci+1):P/(kp*Ci);
    if (p==="pointwise") return bias?P/(Ci+1):P/Ci;
    if (p==="separable") return (P-kp*Ci-(bias?Ci:0))/(Ci+(bias?1:0));
  }
  return NaN;
}
function solveFor(t, vals, target, dims, bias) {
  const cfg = TABS[t];
  if (target === "W_out")  return spatialOut(t, vals, dims);
  if (target === "params") return paramCount(t, vals, dims, bias);
  if (target === "C_in" || target === "C_out")
    return verifyRound(guessChannel(t, vals, dims, bias, target), c => paramCount(t, {...vals, [target]:c}, dims, bias) === vals.params);
  const K = Kof(t,vals), D = cfg.dilated ? vals.D : 1, P = Pof(t,vals), S = vals.S, Wout = vals.W_out;
  if (cfg.spatial === "transposed") {
    if (target==="W_in") return (Wout+2*P-K)/S+1;
    if (target==="K")    return Wout-(vals.W_in-1)*S+2*P;
    if (target==="P")    return ((vals.W_in-1)*S+K-Wout)/2;
    if (target==="S")    return (Wout+2*P-K)/(vals.W_in-1);
  } else {
    const Keff = D*(K-1)+1;
    if (target==="W_in") return (Wout-1)*S - 2*P + Keff;
    if (target==="K") return verifyRound((vals.W_in+2*P-(Wout-1)*S-1)/D+1, c=>spatialOut(t,{...vals,K:c},dims)===Wout);
    if (target==="P") return verifyRound(((Wout-1)*S-vals.W_in+Keff)/2,    c=>spatialOut(t,{...vals,P:c},dims)===Wout);
    if (target==="S") return verifyRound((vals.W_in+2*P-Keff)/(Wout-1),    c=>spatialOut(t,{...vals,S:c},dims)===Wout);
    if (target==="D") return verifyRound((vals.W_in+2*P-(Wout-1)*S-1)/(vals.K-1), c=>spatialOut(t,{...vals,D:c},dims)===Wout);
  }
  return null;
}
function grade(t, vals, hidden, userVal, dims, bias) {
  if (hidden === "W_out")  return userVal === spatialOut(t, vals, dims);
  if (hidden === "params") return userVal === paramCount(t, vals, dims, bias);
  const sub = {...vals, [hidden]: userVal};
  let ok = spatialOut(t, sub, dims) === vals.W_out;
  if (hasParams(t)) ok = ok && paramCount(t, sub, dims, bias) === vals.params;
  return ok;
}
function eligible(t) {
  const cfg = TABS[t], f = ["W_out"];
  if (hasParams(t)) { f.push("params","C_in"); if (cfg.channels==="free") f.push("C_out"); }
  f.push("W_in","S");
  if (cfg.k!=="one") f.push("K");
  if (cfg.hasP) f.push("P");
  if (cfg.dilated) f.push("D");
  return f;
}
function uniqueHidden(t, v, hidden, dims, bias) {
  if (["W_out","params","C_in","C_out"].includes(hidden)) return true;
  const MAX = {W_in:80, K:15, P:12, S:20, D:10}[hidden] || 80;
  let count=0;
  for (let c=1;c<=MAX;c++){ if (grade(t, v, hidden, c, dims, bias)) { count++; if(count>1) return false; } }
  return count===1;
}
const rand=(a,b)=>Math.floor(Math.random()*(b-a+1))+a;
function gen(t, dims, bias, forced, depth=0) {
  if (depth > 200) return null;
  const cfg = TABS[t]; let v = { N: rand(1,8), C_in: rand(1,16)*4 };
  if (cfg.channels==="free") v.C_out = rand(1,16)*4;
  v.W_in = rand(8, 32);
  v.K = cfg.k==="one" ? 1 : (dims===3?rand(1,3)*2-1:rand(1,7)*2-1);
  if (cfg.dilated) v.D = rand(1,3);
  v.P = cfg.hasP ? rand(0, Math.floor((cfg.k==="one"?1:v.K)/2)) : 0;
  v.S = cfg.spatial==="transposed" ? rand(1,3) : rand(1, Math.min(3, v.K + 2*v.P));
  v.W_out = spatialOut(t, v, dims);
  v.params = paramCount(t, v, dims, bias);
  if (v.W_out==null || v.W_out<1) return gen(t,dims,bias,forced,depth+1);
  const elig = eligible(t);
  const hidden = forced || elig[rand(0,elig.length-1)];
  const ans = solveFor(t, v, hidden, dims, bias);
  if (ans==null || ans<=0 || Math.abs(ans-Math.round(ans))>1e-6) return gen(t,dims,bias,forced,depth+1);
  if (!uniqueHidden(t, v, hidden, dims, bias)) return gen(t,dims,bias,forced,depth+1);
  return { v, hidden, mode:t };
}

// ── Working renderers ─────────────────────────────────────────────────────────
const blk = c => ({ color:c, display:"block" });
function SpatialWork({ t, v, dims, result, accent }) {
  const cfg = TABS[t], K = Kof(t,v), D = cfg.dilated ? v.D : 1, P = Pof(t,v), S = v.S, W = v.W_in;
  const s = blk("#888"), r = blk(accent);
  if (cfg.spatial === "transposed") return <>
    <span style={s}>W_out = (W_in − 1)×S − 2P + K</span>
    <span style={s}>= ({W}−1)×{S} − 2×{P} + {K}</span>
    <span style={r}>= {result}</span></>;
  const Keff = D*(K-1)+1;
  return <>
    {cfg.dilated && <span style={s}>K_eff = D(K−1)+1 = {D}×({K}−1)+1 = {Keff}</span>}
    <span style={s}>W_out = ⌊(W_in + 2P − {cfg.dilated?"K_eff":"K"}) / S⌋ + 1</span>
    <span style={s}>= ⌊({W} + 2×{P} − {Keff}) / {S}⌋ + 1</span>
    <span style={s}>= ⌊{W + 2*P - Keff} / {S}⌋ + 1</span>
    <span style={r}>= {result}</span></>;
}
function ParamWork({ t, v, dims, bias, result, accent }) {
  const cfg = TABS[t], K = Kof(t,v), Ci = v.C_in, Co = Coof(t,v), kp = ipow(K,dims);
  const s = blk("#888"), r = blk(accent);
  if (cfg.params === "none") return <span style={r}>Pooling has no learnable parameters → 0</span>;
  if (cfg.params === "regular") return <>
    <span style={s}>#params = K{sup(dims)} × C_in × C_out{bias?" + C_out":""}</span>
    <span style={s}>= {K}{sup(dims)} × {Ci} × {Co}{bias?` + ${Co}`:""}</span>
    <span style={s}>= {kp} × {Ci} × {Co}{bias?` + ${Co}`:""}</span>
    <span style={r}>= {result}</span></>;
  if (cfg.params === "depthwise") return <>
    <span style={s}>one K{sup(dims)} filter per input channel</span>
    <span style={s}>#params = K{sup(dims)} × C_in{bias?" + C_in":""}</span>
    <span style={s}>= {kp} × {Ci}{bias?` + ${Ci}`:""}</span>
    <span style={r}>= {result}</span></>;
  if (cfg.params === "pointwise") return <>
    <span style={s}>1×1 conv mixing channels</span>
    <span style={s}>#params = C_in × C_out{bias?" + C_out":""}</span>
    <span style={s}>= {Ci} × {Co}{bias?` + ${Co}`:""}</span>
    <span style={r}>= {result}</span></>;
  // separable
  const dw = kp*Ci, pw = Ci*Co;
  return <>
    <span style={s}>Depth-wise: K{sup(dims)} × C_in = {kp} × {Ci} = {dw}{bias?` (+${Ci} bias)`:""}</span>
    <span style={s}>Point-wise: C_in × C_out = {Ci} × {Co} = {pw}{bias?` (+${Co} bias)`:""}</span>
    <span style={s}>Total = {dw} + {pw}{bias?` + ${Ci} + ${Co}`:""}</span>
    <span style={r}>= {result}</span></>;
}

// ── Explorer defaults ─────────────────────────────────────────────────────────
const EX_DEFAULTS = {
  standard:   { N:1, C_in:3,  C_out:16, W_in:28, K:3, P:0, S:1 },
  dilated:    { N:1, C_in:3,  C_out:16, W_in:28, K:3, D:2, P:0, S:1 },
  transposed: { N:1, C_in:16, C_out:8,  W_in:4,  K:3, P:0, S:2 },
  pooling:    { N:1, C_in:16, W_in:28, K:2, S:2 },
  depthwise:  { N:1, C_in:16, W_in:28, K:3, P:1, S:1 },
  pointwise:  { N:1, C_in:16, C_out:64, W_in:28, K:1, S:1 },
  separable:  { N:1, C_in:16, C_out:64, W_in:28, K:3, P:1, S:1 },
};

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [appMode, setAppMode] = useState("quiz");
  const [tab, setTab]         = useState("standard");
  const [dims, setDims]       = useState(2);
  const [bias, setBias]       = useState(false);

  const [problem, setProblem] = useState(() => gen("standard", 2, false, null));
  const [input, setInput]     = useState("");
  const [status, setStatus]   = useState(null);
  const [streak, setStreak]   = useState(0);
  const [best, setBest]       = useState(0);
  const [total, setTotal]     = useState(0);
  const [correct, setCorrect] = useState(0);
  const [showHint, setShowHint] = useState(false);
  const inputRef = useRef(null);

  const [exVals, setExVals] = useState({ ...EX_DEFAULTS["standard"] });

  useEffect(() => { if (appMode==="quiz" && inputRef.current) inputRef.current.focus(); }, [problem, appMode]);

  function newProblem(t=tab, forced=null, d=dims, b=bias) { setProblem(gen(t, d, b, forced)); setInput(""); setStatus(null); setShowHint(false); }
  function changeTab(t) {
    setTab(t);
    if (appMode==="quiz") newProblem(t);
    else setExVals({ ...EX_DEFAULTS[t] });
    setStreak(0);
  }
  function switchApp(m) { setAppMode(m); if (m==="explorer") setExVals({ ...EX_DEFAULTS[tab] }); else newProblem(tab); }
  function changeDims(d) { setDims(d); if (appMode==="quiz") newProblem(tab, null, d, bias); }
  function changeBias(b) { setBias(b); if (appMode==="quiz") newProblem(tab, null, dims, b); }

  function submit() {
    if (status==="correct") { newProblem(); return; }
    if (!input.trim()) return;
    const ok = grade(problem.mode, problem.v, problem.hidden, parseFloat(input.trim()), dims, bias);
    setTotal(x=>x+1);
    if (ok) { setStatus("correct"); setCorrect(x=>x+1); const ns=streak+1; setStreak(ns); setBest(b=>Math.max(b,ns)); }
    else    { setStatus("wrong"); setStreak(0); }
  }
  function onKey(e){ if(e.key==="Enter") submit(); }
  function pinQuiz(f){ if (status!=="correct" && eligible(tab).includes(f)) newProblem(tab, f); }
  function exChange(f, raw){ setExVals(p => ({ ...p, [f]: raw===""?"":Number(raw) })); }

  const accent = modeColors[tab];
  const cfg = TABS[tab];
  const vis = visibleFields(tab);
  const elig = eligible(tab);

  // quiz derived
  const { v, hidden } = problem || { v:{}, hidden:null };
  const genAnswer = problem ? (hidden==="W_out"?v.W_out : hidden==="params"?v.params : v[hidden]) : null;
  const quizWout = problem ? spatialOut(tab, v, dims) : null;
  const quizCo = problem ? Coof(tab, v) : null;

  // explorer derived
  const exWout = spatialOut(tab, exVals, dims);
  const exParams = paramCount(tab, exVals, dims, bias);
  const exCo = Coof(tab, exVals);

  const dispVal = (src, f) => f==="C_out" && cfg.channels==="locked" ? src.C_in : f==="K" && cfg.k==="one" ? 1 : src[f];
  const shapeStr = (N, Co, W) => `${N ?? "?"} × ${Co ?? "?"} × ${W!=null ? Array(dims).fill(W).join(" × ") : "?"}`;
  const paramHint = hidden && ["C_in","C_out","params"].includes(hidden);

  return (
    <div style={{ fontFamily:"'IBM Plex Mono', monospace", minHeight:"100vh", background:"#0d0d0f", color:"#e8e4dc" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        body { background:#0d0d0f; }
        .tab-btn { font-family:'IBM Plex Sans',sans-serif; font-size:12px; font-weight:500; letter-spacing:.06em; padding:7px 18px; border-radius:3px; border:1px solid #2a2a2e; background:transparent; color:#555; cursor:pointer; transition:all .15s; text-transform:uppercase; }
        .tab-btn:hover { color:#999; border-color:#444; }
        .tab-btn.on { color:#0d0d0f; border-color:transparent; }
        .mode-btn { font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:500; letter-spacing:.05em; padding:6px 11px; border-radius:3px; border:1px solid #2a2a2e; background:transparent; color:#666; cursor:pointer; transition:all .15s; text-transform:uppercase; }
        .mode-btn:hover { border-color:#444; color:#aaa; }
        .mode-btn.on { color:#0d0d0f; border-color:transparent; }
        .seg { font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; padding:5px 12px; border:1px solid #2a2a2e; background:transparent; color:#666; cursor:pointer; transition:all .12s; }
        .seg:first-child{border-radius:3px 0 0 3px;} .seg:last-child{border-radius:0 3px 3px 0;} .seg+.seg{border-left:none;}
        .seg.on{color:#0d0d0f;}
        .toggle{ display:inline-flex; align-items:center; gap:8px; cursor:pointer; font-family:'IBM Plex Sans',sans-serif; font-size:12px; color:#888; user-select:none; }
        .tk{ width:34px; height:18px; border-radius:9px; background:#2a2a2e; position:relative; transition:background .15s; }
        .tk.on{ background:var(--ac); } .tknob{ position:absolute; top:2px; left:2px; width:14px; height:14px; border-radius:50%; background:#e8e4dc; transition:transform .15s; } .tk.on .tknob{ transform:translateX(16px); }
        .row{ display:flex; align-items:center; gap:12px; padding:9px 16px; border-bottom:1px solid #1a1a1e; transition:background .12s; position:relative; }
        .row:last-child{ border-bottom:none; }
        .row.hide{ background:#16161a; } .row.comp{ background:#0a1a14; } .row.lock{ opacity:.55; }
        .row.pin{ cursor:pointer; } .row.pin:hover{ background:#141418; } .row.pin:hover .ph{ opacity:1; }
        .ph{ opacity:0; font-size:10px; color:#444; font-family:'IBM Plex Sans',sans-serif; transition:opacity .15s; margin-left:auto; white-space:nowrap; letter-spacing:.06em; }
        .lbl{ font-size:11px; color:#555; width:130px; flex-shrink:0; font-family:'IBM Plex Sans',sans-serif; letter-spacing:.04em; }
        .sym{ font-size:13px; font-weight:600; width:62px; flex-shrink:0; }
        .val{ font-size:19px; font-weight:600; }
        .ai{ font-family:'IBM Plex Mono',monospace; font-size:19px; font-weight:600; background:transparent; border:none; border-bottom:2px solid; color:#e8e4dc; outline:none; width:130px; padding:2px 0; }
        .ei{ font-family:'IBM Plex Mono',monospace; font-size:19px; font-weight:600; background:transparent; border:none; border-bottom:1px solid #2a2a2e; color:#e8e4dc; outline:none; width:130px; padding:2px 0; transition:border-color .15s; }
        .ei:focus{ border-bottom-color:#888; }
        .btn{ font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; padding:12px 28px; border-radius:3px; cursor:pointer; border:none; transition:all .15s; width:100%; }
        .btn:hover{ filter:brightness(1.1); transform:translateY(-1px); }
        .skip{ font-family:'IBM Plex Mono',monospace; font-size:11px; background:transparent; border:1px solid #222; color:#444; padding:8px 16px; border-radius:3px; cursor:pointer; letter-spacing:.06em; text-transform:uppercase; transition:all .15s; width:100%; }
        .skip:hover{ border-color:#444; color:#666; }
        .stat{ text-align:center; } .stat-n{ font-size:26px; font-weight:600; line-height:1; } .stat-l{ font-size:10px; color:#444; letter-spacing:.1em; text-transform:uppercase; margin-top:4px; font-family:'IBM Plex Sans',sans-serif; }
        .fbox{ background:#111115; border:1px solid #222226; border-radius:4px; padding:10px 16px; font-size:12px; color:#888; letter-spacing:.02em; }
        .badge{ display:inline-block; font-size:10px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; padding:3px 8px; border-radius:2px; font-family:'IBM Plex Sans',sans-serif; }
        .card{ border:1px solid #1e1e24; border-radius:6px; overflow:hidden; background:#0f0f13; }
        .chd{ padding:12px 16px; border-bottom:1px solid #1a1a1e; display:flex; align-items:center; justify-content:space-between; }
        .chl{ font-size:10px; color:#444; letter-spacing:.14em; text-transform:uppercase; font-family:'IBM Plex Sans',sans-serif; }
      `}</style>

      <div style={{ maxWidth:560, margin:"0 auto", padding:"32px 20px" }}>

        {/* Header */}
        <div style={{ marginBottom:22 }}>
          <div style={{ display:"flex", alignItems:"baseline", gap:10, marginBottom:6 }}>
            <span style={{ fontSize:11, color:"#444", letterSpacing:".16em", textTransform:"uppercase", fontFamily:"'IBM Plex Sans',sans-serif" }}>MCEN90048</span>
            <span style={{ color:"#222" }}>—</span>
            <span style={{ fontSize:11, color:"#444", letterSpacing:".16em", textTransform:"uppercase", fontFamily:"'IBM Plex Sans',sans-serif" }}>CNN Drill</span>
          </div>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
            <h1 style={{ fontSize:22, fontWeight:600, letterSpacing:"-.02em", fontFamily:"'IBM Plex Sans',sans-serif" }}>{appMode==="quiz"?"Find the missing value":"Explorer"}</h1>
            <div style={{ display:"flex", gap:4, background:"#111115", border:"1px solid #1e1e24", borderRadius:4, padding:3 }}>
              {["quiz","explorer"].map(m => <button key={m} className={`tab-btn${appMode===m?" on":""}`} style={appMode===m?{backgroundColor:accent}:{}} onClick={()=>switchApp(m)}>{m}</button>)}
            </div>
          </div>
        </div>

        {/* Stats */}
        {appMode==="quiz" && (
          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:1, background:"#1a1a1e", borderRadius:6, padding:1, marginBottom:22 }}>
            {[{n:streak,l:"Streak",c:streak>0?accent:"#e8e4dc"},{n:best,l:"Best"},{n:correct,l:"Correct"},{n:total>0?Math.round(correct/total*100)+"%":"—",l:"Accuracy"}].map(s=>(
              <div key={s.l} className="stat" style={{ background:"#0d0d0f", padding:"12px 8px", borderRadius:5 }}>
                <div className="stat-n" style={{ color:s.c||"#e8e4dc" }}>{s.n}</div><div className="stat-l">{s.l}</div>
              </div>
            ))}
          </div>
        )}

        {/* Tabs */}
        <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:16 }}>
          {TAB_IDS.map(t => <button key={t} className={`mode-btn${tab===t?" on":""}`} style={tab===t?{backgroundColor:modeColors[t]}:{}} onClick={()=>changeTab(t)}>{TABS[t].label}</button>)}
        </div>

        {/* Settings: dims + bias */}
        <div style={{ display:"flex", alignItems:"center", gap:18, marginBottom:16, padding:"10px 14px", background:"#111115", border:"1px solid #1e1e24", borderRadius:4, flexWrap:"wrap" }}>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <span style={{ fontSize:11, color:"#555", fontFamily:"'IBM Plex Sans',sans-serif", letterSpacing:".06em" }}>DIMS</span>
            <div style={{ display:"flex" }}>{[1,2,3].map(d=><button key={d} className={`seg${dims===d?" on":""}`} style={dims===d?{backgroundColor:accent,borderColor:accent}:{}} onClick={()=>changeDims(d)}>{d}D</button>)}</div>
          </div>
          {hasParams(tab) && (
            <label className="toggle" style={{ "--ac":accent }} onClick={()=>changeBias(!bias)}>
              <span className={`tk${bias?" on":""}`}><span className="tknob"/></span><span>include bias</span>
            </label>
          )}
          {!hasParams(tab) && <span style={{ fontSize:11, color:"#444", fontFamily:"'IBM Plex Sans',sans-serif" }}>pooling has no parameters</span>}
        </div>

        {/* Formula */}
        <div className="fbox" style={{ marginBottom:20 }}>
          {cfg.spatial==="transposed" ? "W_out = (W_in − 1)×S − 2P + K" :
           cfg.dilated ? "W_out = ⌊(W_in + 2P − [D(K−1)+1]) / S⌋ + 1" :
           "W_out = ⌊(W_in + 2P − K) / S⌋ + 1"}
          {hasParams(tab) && <span style={{ color:"#555" }}>{"   ·   #params = "}{
            cfg.params==="regular" ? `K${sup(dims)}·C_in·C_out${bias?"+C_out":""}` :
            cfg.params==="depthwise" ? `K${sup(dims)}·C_in${bias?"+C_in":""}` :
            cfg.params==="pointwise" ? `C_in·C_out${bias?"+C_out":""}` :
            `K${sup(dims)}·C_in + C_in·C_out${bias?"+C_in+C_out":""}`
          }</span>}
        </div>

        {/* ══ QUIZ ══ */}
        {appMode==="quiz" && problem && (
          <>
            <div className="card" style={{ marginBottom:20 }}>
              <div className="chd">
                <span className="chl">{cfg.label} · {dims}D{hasParams(tab)&&bias?" · +bias":""}</span>
                <span className="badge" style={{ background:accent+"22", color:accent }}>Solve for {SYMBOLS[hidden]}</span>
              </div>
              {vis.map(f => {
                const isHidden = f===hidden, locked = isLocked(tab,f), canPin = elig.includes(f) && !isHidden;
                return (
                  <div key={f} className={`row${isHidden?" hide":""}${locked?" lock":""}${canPin?" pin":""}`} onClick={()=>canPin&&pinQuiz(f)} title={canPin?`Click to solve for ${SYMBOLS[f]} instead`:""}>
                    <span className="lbl">{LABELS[f]}</span>
                    <span className="sym" style={{ color:isHidden?accent:"#666" }}>{SYMBOLS[f]}</span>
                    {isHidden ? (
                      <div style={{ display:"flex", alignItems:"center", gap:10 }} onClick={e=>e.stopPropagation()}>
                        <input ref={inputRef} className="ai" style={{ borderBottomColor:status==="correct"?"#10b981":status==="wrong"?"#ef4444":accent }} type="number" value={input} placeholder="?" disabled={status==="correct"} onChange={e=>{setInput(e.target.value);setStatus(null);}} onKeyDown={onKey}/>
                        {status==="wrong"   && <span style={{ fontSize:11, color:"#ef4444", fontFamily:"'IBM Plex Sans',sans-serif" }}>✗ got {input}</span>}
                        {status==="correct" && <span style={{ fontSize:11, color:"#10b981", fontFamily:"'IBM Plex Sans',sans-serif" }}>✓ correct</span>}
                      </div>
                    ) : (<><span className="val" style={{ color:(f==="W_out"||f==="params")?"#8a8a92":"#e8e4dc" }}>{dispVal(v,f)}{locked && f==="C_out" && <span style={{ fontSize:11, color:"#444", marginLeft:8 }}>= C_in</span>}{locked && f==="K" && <span style={{ fontSize:11, color:"#444", marginLeft:8 }}>fixed (1×1)</span>}</span>{canPin && <span className="ph">solve for this →</span>}</>)}
                  </div>
                );
              })}
              {/* output shape */}
              <div style={{ padding:"10px 16px", borderTop:"1px dashed #1e1e24", fontSize:12, color:"#555", fontFamily:"'IBM Plex Sans',sans-serif" }}>
                Output tensor: <span style={{ fontFamily:"'IBM Plex Mono',monospace", color:"#777" }}>{shapeStr(v.N, quizCo, hidden==="W_out"?null:quizWout)}</span> &nbsp;(N × C_out × {Array(dims).fill("·").join(" × ")})
              </div>
            </div>

            {status==="wrong" && (
              <div style={{ background:"#1a0a0a", border:"1px solid #3a1515", borderRadius:6, padding:"12px 16px", marginBottom:16 }}>
                <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:showHint?8:0 }}>
                  <span style={{ fontSize:12, color:"#ef4444", fontWeight:500, fontFamily:"'IBM Plex Sans',sans-serif" }}>Answer: {genAnswer}</span>
                  <button style={{ background:"transparent", border:"none", color:"#555", fontSize:11, cursor:"pointer", fontFamily:"'IBM Plex Sans',sans-serif", letterSpacing:".06em" }} onClick={()=>setShowHint(h=>!h)}>{showHint?"hide hint":"show hint"}</button>
                </div>
                {showHint && <div style={{ fontSize:12, color:"#666", lineHeight:2 }}>{paramHint ? <ParamWork t={tab} v={v} dims={dims} bias={bias} result={hidden==="params"?v.params:v.params} accent="#9a5555"/> : <SpatialWork t={tab} v={v} dims={dims} result={hidden==="W_out"?v.W_out:v.W_out} accent="#9a5555"/>}</div>}
              </div>
            )}

            <div style={{ display:"flex", gap:8 }}>
              <button className="btn" style={{ background:status==="correct"?"#10b981":accent, color:"#fff", flex:2 }} onClick={submit}>{status==="correct"?"Next →":"Check"}</button>
              {status!=="correct" && <button className="skip" style={{ flex:1 }} onClick={()=>newProblem()}>Skip</button>}
            </div>

            {status==="correct" && streak>=3 && (
              <div style={{ textAlign:"center", marginTop:20, padding:"10px", background:accent+"11", borderRadius:4 }}>
                <span style={{ fontSize:13, color:accent, fontWeight:600 }}>{streak>=10?"🔥 ":""}{streak} in a row</span>
              </div>
            )}
          </>
        )}

        {/* ══ EXPLORER ══ */}
        {appMode==="explorer" && (
          <>
            <div className="card" style={{ marginBottom:20 }}>
              <div className="chd"><span className="chl">{cfg.label} · {dims}D{hasParams(tab)&&bias?" · +bias":""}</span><span className="badge" style={{ background:accent+"22", color:accent }}>live calculator</span></div>
              {vis.map(f => {
                const computed = f==="W_out" || f==="params", locked = isLocked(tab,f);
                if (computed) {
                  const val = f==="W_out" ? exWout : exParams;
                  return (
                    <div key={f} className="row comp">
                      <span className="lbl" style={{ color:"#2a6650" }}>{LABELS[f]}</span>
                      <span className="sym" style={{ color:"#059669" }}>{SYMBOLS[f]}</span>
                      <span style={{ fontSize:26, fontWeight:600, color: val!=null?accent:"#333" }}>{val!=null?val:"—"}</span>
                    </div>
                  );
                }
                if (locked) {
                  return (
                    <div key={f} className="row lock">
                      <span className="lbl">{LABELS[f]}</span><span className="sym" style={{ color:"#666" }}>{SYMBOLS[f]}</span>
                      <span className="val">{dispVal(exVals,f)}</span>
                      <span style={{ fontSize:11, color:"#444", marginLeft:8 }}>{f==="C_out"?"= C_in":"fixed (1×1)"}</span>
                    </div>
                  );
                }
                return (
                  <div key={f} className="row">
                    <span className="lbl">{LABELS[f]}</span><span className="sym" style={{ color:"#666" }}>{SYMBOLS[f]}</span>
                    <input className="ei" type="number" min="0" value={exVals[f]==null?"":exVals[f]} onChange={e=>exChange(f, e.target.value)}/>
                  </div>
                );
              })}
              <div style={{ padding:"12px 16px", borderTop:"1px dashed #1e1e24", fontSize:13, lineHeight:2, fontFamily:"'IBM Plex Mono',monospace" }}>
                <div style={{ color:"#666" }}>Input&nbsp;&nbsp;= {shapeStr(exVals.N, exVals.C_in, exVals.W_in)}</div>
                <div style={{ color:accent }}>Output = {shapeStr(exVals.N, exCo, exWout)}</div>
                <div style={{ color:"#444", fontSize:11, marginTop:6, fontFamily:"'IBM Plex Sans',sans-serif" }}>Batch N and channels pass through; the spatial formula applies to each of the {dims} axis{dims>1?"es":""}. Batch never affects #params.</div>
              </div>
            </div>

            {(exWout!=null) && (
              <div style={{ background:"#111115", border:"1px solid #1e1e24", borderRadius:6, padding:"14px 16px", marginBottom:14 }}>
                <div style={{ fontSize:10, color:"#444", letterSpacing:".12em", textTransform:"uppercase", fontFamily:"'IBM Plex Sans',sans-serif", marginBottom:10 }}>Spatial working</div>
                <div style={{ fontSize:12, lineHeight:2 }}><SpatialWork t={tab} v={exVals} dims={dims} result={exWout} accent={accent}/></div>
              </div>
            )}
            {hasParams(tab) && exParams!=null && (
              <div style={{ background:"#111115", border:"1px solid #1e1e24", borderRadius:6, padding:"14px 16px" }}>
                <div style={{ fontSize:10, color:"#444", letterSpacing:".12em", textTransform:"uppercase", fontFamily:"'IBM Plex Sans',sans-serif", marginBottom:10 }}>Parameter working</div>
                <div style={{ fontSize:12, lineHeight:2 }}><ParamWork t={tab} v={exVals} dims={dims} bias={bias} result={exParams} accent={accent}/></div>
              </div>
            )}
          </>
        )}

      </div>
    </div>
  );
}
