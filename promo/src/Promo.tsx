import React from "react";
import { AbsoluteFill, interpolate, Series, useCurrentFrame } from "remotion";
import { C } from "./theme";
import { Badge, Chip, display, Eyebrow, Grid, mono, rise, useReveal, Vignette, Wordmark } from "./ui";

const D = { title: 90, concept: 170, local: 90, engines: 160, formats: 90, outro: 110 };
export const PROMO_DURATION = Object.values(D).reduce((a, b) => a + b, 0);

const clamp = { extrapolateLeft: "clamp" as const, extrapolateRight: "clamp" as const };

// Fades each scene in over 8 frames so the Series cuts read as soft dissolves
// against the persistent dot-grid ground.
const Scene: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 8], [0, 1], clamp);
  return (
    <AbsoluteFill style={{ opacity, justifyContent: "center", alignItems: "center" }}>
      {children}
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- Title
const Title: React.FC = () => {
  const frame = useCurrentFrame();
  const badge = useReveal(4);
  return (
    <Scene>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 30 }}>
        <div style={{ opacity: badge, transform: `scale(${interpolate(badge, [0, 1], [0.6, 1])})`, display: "flex", alignItems: "center", gap: 18 }}>
          <Badge size={40} />
          <span style={{ fontFamily: mono, fontSize: 30, color: C.inkSoft, letterSpacing: "0.06em" }}>local-llm-pdf-ocr</span>
        </div>
        <h1 style={{ ...rise(frame, 26, 22), fontFamily: display, fontWeight: 800, fontSize: 96, color: C.ink, margin: 0, letterSpacing: "-0.025em", textAlign: "center", lineHeight: 1.02 }}>
          Searchable PDFs<br />from scans.
        </h1>
        <div style={rise(frame, 48, 22)}>
          <Eyebrow>100% local vision-LLM OCR · nothing leaves your machine</Eyebrow>
        </div>
      </div>
    </Scene>
  );
};

// ---------------------------------------------------------------- Concept
const LINES = [0.92, 0.7, 0.82, 0.5, 0.88, 0.64, 0.78, 0.42, 0.72, 0.6];
const Concept: React.FC = () => {
  const frame = useCurrentFrame();
  const pageW = 520, pageH = 620, pad = 48, gap = 22, lineH = 16;
  const beam = interpolate(frame, [22, 84], [0, 1], clamp);
  const beamPx = beam * pageH;
  return (
    <Scene>
      <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 76 }}>
        <div style={{ position: "relative", width: pageW, height: pageH, background: C.surface, border: `1px solid ${C.hairline}`, borderRadius: 2, overflow: "hidden", ...rise(frame, 4, 18, 30) }}>
          <div style={{ position: "absolute", left: pad, right: pad, top: pad, display: "flex", flexDirection: "column", gap }}>
            {LINES.map((w, i) => {
              const y = pad + i * (lineH + gap);
              const seen = beamPx > y;
              const sel = interpolate(frame, [86 + i * 5, 98 + i * 5], [0, 1], clamp);
              return (
                <div key={i} style={{ position: "relative", height: lineH }}>
                  <div style={{ width: `${w * 100}%`, height: lineH, borderRadius: 2, background: seen ? C.hairlineStrong : C.hairline, opacity: seen ? 1 : 0.55 }} />
                  <div style={{ position: "absolute", top: -3, left: -4, height: lineH + 6, width: `calc(${w * 100}% + 8px)`, background: C.accent, opacity: sel * 0.3, borderRadius: 2 }} />
                </div>
              );
            })}
          </div>
          {frame < 88 && (
            <div style={{ position: "absolute", left: 0, right: 0, top: beamPx, height: 3, background: C.accent, boxShadow: `0 0 28px 7px ${C.accent}` }} />
          )}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 20, width: 380, ...rise(frame, 92, 20) }}>
          <Eyebrow color={C.accent}>image in → text out</Eyebrow>
          <div style={{ fontFamily: display, fontWeight: 700, fontSize: 46, color: C.ink, lineHeight: 1.1, letterSpacing: "-0.01em" }}>An invisible text layer under the original page.</div>
          <div style={{ fontFamily: mono, fontSize: 21, color: C.inkSoft, lineHeight: 1.55 }}>Select it. Search it. Copy it. The scan itself looks untouched.</div>
        </div>
      </div>
    </Scene>
  );
};

// ---------------------------------------------------------------- Local
const Local: React.FC = () => {
  const frame = useCurrentFrame();
  const chips = ["no cloud", "no api keys", "nothing uploaded"];
  return (
    <Scene>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 34, textAlign: "center" }}>
        <div style={rise(frame, 4, 18)}><Eyebrow color={C.ok}>● runs entirely on this machine</Eyebrow></div>
        <h2 style={{ ...rise(frame, 16, 20), fontFamily: display, fontWeight: 800, fontSize: 80, color: C.ink, margin: 0, letterSpacing: "-0.025em", lineHeight: 1.02 }}>
          Your documents<br />never leave.
        </h2>
        <div style={{ display: "flex", gap: 16 }}>
          {chips.map((c, i) => (<div key={c} style={rise(frame, 36 + i * 8, 16)}><Chip>{c}</Chip></div>))}
        </div>
      </div>
    </Scene>
  );
};

// ---------------------------------------------------------------- Engines
const ENGINES = [
  { k: "Hybrid", d: "Surya layout + a full-page vision LLM + DP line-to-box alignment.", win: [10, 60] },
  { k: "Grounded", d: "A bbox-native VLM returns text and coordinates in one single call.", win: [60, 108] },
  { k: "Text only", d: "Full-page plain-text dump — no layout model, the fastest path.", win: [108, 160] },
];
const Engines: React.FC = () => {
  const frame = useCurrentFrame();
  const active = frame < 60 ? 0 : frame < 108 ? 1 : 2;
  return (
    <Scene>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 44 }}>
        <div style={rise(frame, 2, 16)}><Eyebrow>one tool · three engines</Eyebrow></div>
        <div style={{ display: "inline-flex", border: `1px solid ${C.hairline}`, borderRadius: 2, overflow: "hidden", ...rise(frame, 10, 16) }}>
          {ENGINES.map((e, i) => (
            <div key={e.k} style={{ fontFamily: mono, fontSize: 30, textTransform: "uppercase", letterSpacing: "0.06em", padding: "20px 46px", borderRight: i < 2 ? `1px solid ${C.hairline}` : "none", background: active === i ? C.accent : C.surface, color: active === i ? C.accentContrast : C.inkSoft }}>
              {e.k}
            </div>
          ))}
        </div>
        <div style={{ position: "relative", height: 96, width: 880, textAlign: "center" }}>
          {ENGINES.map((e, i) => (
            <div key={e.k} style={{ position: "absolute", inset: 0, opacity: active === i ? interpolate(frame, [e.win[0], e.win[0] + 10], [0, 1], clamp) : 0, fontFamily: display, fontWeight: 600, fontSize: 38, color: C.ink, lineHeight: 1.3 }}>
              {e.d}
            </div>
          ))}
        </div>
      </div>
    </Scene>
  );
};

// ---------------------------------------------------------------- Formats
const Formats: React.FC = () => {
  const frame = useCurrentFrame();
  const fmts = ["Searchable PDF", "HTML overlay", "Markdown", "Plain text"];
  return (
    <Scene>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 38 }}>
        <div style={rise(frame, 2, 16)}><Eyebrow>four ways out</Eyebrow></div>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap", justifyContent: "center", maxWidth: 1040 }}>
          {fmts.map((f, i) => (<div key={f} style={rise(frame, 12 + i * 9, 16)}><Chip on={i === 0}>{f}</Chip></div>))}
        </div>
      </div>
    </Scene>
  );
};

// ---------------------------------------------------------------- Outro
const CMD = "uv run local-llm-pdf-ocr scan.pdf";
const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const chars = Math.round(interpolate(frame, [10, 58], [0, CMD.length], clamp));
  const caretOn = Math.floor(frame / 16) % 2 === 0;
  return (
    <Scene>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 42 }}>
        <div style={{ ...rise(frame, 2, 16), background: C.surface, border: `1px solid ${C.hairline}`, borderRadius: 2, padding: "26px 36px", minWidth: 760 }}>
          <span style={{ fontFamily: mono, fontSize: 32, color: C.ok }}>$ </span>
          <span style={{ fontFamily: mono, fontSize: 32, color: C.ink }}>{CMD.slice(0, chars)}</span>
          <span style={{ fontFamily: mono, fontSize: 32, color: C.accent, opacity: caretOn ? 1 : 0 }}>▋</span>
        </div>
        <div style={rise(frame, 62, 18)}><Wordmark size={36} /></div>
        <div style={rise(frame, 72, 18)}>
          <span style={{ fontFamily: mono, fontSize: 23, color: C.inkSoft, letterSpacing: "0.04em" }}>github.com/ahnafnafee/local-llm-pdf-ocr</span>
        </div>
      </div>
    </Scene>
  );
};

// ---------------------------------------------------------------- Composition
export const Promo: React.FC = () => (
  <AbsoluteFill>
    <Grid />
    <Series>
      <Series.Sequence durationInFrames={D.title}><Title /></Series.Sequence>
      <Series.Sequence durationInFrames={D.concept}><Concept /></Series.Sequence>
      <Series.Sequence durationInFrames={D.local}><Local /></Series.Sequence>
      <Series.Sequence durationInFrames={D.engines}><Engines /></Series.Sequence>
      <Series.Sequence durationInFrames={D.formats}><Formats /></Series.Sequence>
      <Series.Sequence durationInFrames={D.outro}><Outro /></Series.Sequence>
    </Series>
    <Vignette />
  </AbsoluteFill>
);
