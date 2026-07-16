import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";
import { loadFont as loadDisplay } from "@remotion/google-fonts/Archivo";
import { C } from "./theme";

// Load only the weights/subset actually used — keeps the render's font
// fetch small and silences the "too many network requests" warning.
export const mono = loadMono("normal", {
  weights: ["400", "500", "700"],
  subsets: ["latin"],
  ignoreTooManyRequestsWarning: true,
}).fontFamily;
export const display = loadDisplay("normal", {
  weights: ["600", "700", "800"],
  subsets: ["latin"],
  ignoreTooManyRequestsWarning: true,
}).fontFamily;

// Smooth, no-overshoot spring 0..1 that starts at `delay` frames.
export const useReveal = (delay = 0, damping = 200): number => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: frame - delay, fps, config: { damping } });
};

// The notebook dot-grid ground, shared behind every scene.
export const Grid: React.FC = () => (
  <AbsoluteFill
    style={{
      backgroundColor: C.bg,
      backgroundImage: `radial-gradient(${C.gridDot} 1.5px, transparent 1.5px)`,
      backgroundSize: "34px 34px",
    }}
  />
);

// A soft darkroom vignette to focus the eye.
export const Vignette: React.FC = () => (
  <AbsoluteFill
    style={{
      background: `radial-gradient(120% 90% at 50% 45%, transparent 55%, rgba(0,0,0,0.45) 100%)`,
    }}
  />
);

export const Badge: React.FC<{ size?: number }> = ({ size = 26 }) => (
  <span
    style={{
      fontFamily: mono,
      fontWeight: 700,
      letterSpacing: "0.14em",
      background: C.accent,
      color: C.accentContrast,
      padding: `${size * 0.22}px ${size * 0.42}px`,
      fontSize: size,
      lineHeight: 1,
    }}
  >
    OCR
  </span>
);

export const Wordmark: React.FC<{ size?: number }> = ({ size = 26 }) => (
  <div style={{ display: "flex", alignItems: "center", gap: size * 0.55 }}>
    <Badge size={size} />
    <span
      style={{
        fontFamily: mono,
        fontSize: size * 0.82,
        letterSpacing: "0.06em",
        color: C.inkSoft,
      }}
    >
      local-llm-pdf-ocr
    </span>
  </div>
);

// A mono pill used for feature/format chips.
export const Chip: React.FC<{ children: React.ReactNode; on?: boolean }> = ({ children, on }) => (
  <span
    style={{
      fontFamily: mono,
      fontSize: 24,
      textTransform: "uppercase",
      letterSpacing: "0.08em",
      color: on ? C.accentContrast : C.ink,
      background: on ? C.accent : C.surface,
      border: `1px solid ${on ? C.accent : C.hairlineStrong}`,
      borderRadius: 2,
      padding: "12px 20px",
    }}
  >
    {children}
  </span>
);

// Uppercase mono eyebrow label.
export const Eyebrow: React.FC<{ children: React.ReactNode; color?: string }> = ({
  children,
  color = C.inkSoft,
}) => (
  <span
    style={{
      fontFamily: mono,
      fontSize: 20,
      textTransform: "uppercase",
      letterSpacing: "0.22em",
      color,
    }}
  >
    {children}
  </span>
);

// Helper: fade+rise a block in over `dur` starting at `delay`.
export const rise = (frame: number, delay: number, dur = 20, dy = 22) => ({
  opacity: interpolate(frame, [delay, delay + dur], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
  transform: `translateY(${interpolate(frame, [delay, delay + dur], [dy, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  })}px)`,
});
