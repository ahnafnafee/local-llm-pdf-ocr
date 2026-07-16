# Promo video (Remotion)

The animated promo used as the README hero, built with [Remotion](https://remotion.dev).
Self-contained Node subproject — it does **not** touch the Python/`uv` build.

The look is the app's own "print-bench" dark theme: darkroom slate, phosphor-amber
accent, monospace instrument labels, notebook dot-grid ground.

## Render

```bash
cd promo
npm install                 # first run also downloads a headless Chromium
npm run render              # -> out/promo.mp4 (1280x720, ~24s)
```

Then produce the optimized README GIF from the mp4 with ffmpeg:

```bash
ffmpeg -i out/promo.mp4 -vf "fps=15,scale=960:-1:flags=lanczos,palettegen=stats_mode=diff" -y out/palette.png
ffmpeg -i out/promo.mp4 -i out/palette.png -lavfi "fps=15,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer" -y ../docs/promo.gif
```

Commit `docs/promo.gif` (README hero) and, if you want the full-quality file,
`docs/promo.mp4`.

## Preview / edit

```bash
npm run studio             # opens the Remotion Studio for live editing
```

## Structure

| File | Role |
|------|------|
| `src/Root.tsx` | Registers the `Promo` composition (1280×720, 30fps) |
| `src/Promo.tsx` | The six scenes (title → concept → local → engines → formats → outro) |
| `src/ui.tsx` | Shared components + Google-fonts (Archivo display, JetBrains Mono) |
| `src/theme.ts` | Print-bench darkroom palette |
