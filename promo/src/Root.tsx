import React from "react";
import { Composition } from "remotion";
import { Promo, PROMO_DURATION } from "./Promo";
import { FPS } from "./theme";

export const RemotionRoot: React.FC = () => (
  <Composition
    id="Promo"
    component={Promo}
    durationInFrames={PROMO_DURATION}
    fps={FPS}
    width={1280}
    height={720}
  />
);
