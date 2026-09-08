/**
 * LiquidGlassDefs — the SVG filters that power the trending iOS-26 / WWDC-2025
 * "Liquid Glass" look. Injected once at app root; used via `backdrop-filter:
 * url(#…)` from the `.lg` classes in index.css.
 *
 * What makes it read as TRENDING liquid glass (not just frosted glass):
 *   1. Edge refraction — a displacement map (radial, neutral 128 in the centre,
 *      drifting at the rim) bends the backdrop like a thick lens.
 *   2. Chromatic aberration — R / G / B are displaced at slightly different
 *      scales then recombined, so the refracted edges fringe red↔blue. This is
 *      the signature "premium glass" tell.
 *   3. Specular gloss — feSpecularLighting adds a bright wet highlight.
 *
 * CHROMIUM-ONLY (SVG filter as backdrop-filter). Tauri Windows = WebView2
 * (Chromium) → works. Elsewhere the `.lg` classes fall back to frosted blur.
 */
import { useEffect } from "react";

// Radial displacement map: centre rgb(128,128,…) = no shift; rim pushes R up and
// G down so the lens bends outward. Reused (at different feDisplacementMap
// scales) for each colour channel to create chromatic aberration.
const DISP_MAP =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' width='300' height='200'>
       <defs>
         <radialGradient id='g' cx='50%' cy='50%' r='72%'>
           <stop offset='0%'   stop-color='rgb(128,128,128)'/>
           <stop offset='48%'  stop-color='rgb(128,128,128)'/>
           <stop offset='100%' stop-color='rgb(210,70,128)'/>
         </radialGradient>
       </defs>
       <rect width='300' height='200' fill='rgb(128,128,128)'/>
       <rect width='300' height='200' rx='48' ry='48' fill='url(#g)'/>
     </svg>`
  );

export function LiquidGlassDefs() {
  useEffect(() => {
    const ua = navigator.userAgent;
    const isChromium = /Chrome|Chromium|Edg|WebView2/.test(ua) && !/Firefox/.test(ua);
    if (isChromium) document.documentElement.classList.add("lg-refract");
  }, []);

  return (
    <svg aria-hidden width="0" height="0"
      style={{ position: "absolute", pointerEvents: "none" }}
      colorInterpolationFilters="sRGB">
      <defs>
        {/* ── Main liquid-glass filter: refraction + chromatic aberration + gloss ── */}
        <filter id="lgFilter" x="-35%" y="-35%" width="170%" height="170%">
          <feImage href={DISP_MAP} x="0" y="0" width="100%" height="100%"
            preserveAspectRatio="none" result="map" />

          {/* Light frost so refracted detail stays readable. */}
          <feGaussianBlur in="SourceGraphic" stdDeviation="0.6" result="src" />

          {/* PERF-TONED: ONE displacement pass instead of three. The per-channel
           * (R/G/B) chromatic-aberration variant sampled the backdrop 3× per pixel
           * per .lg element for a fringe that was barely visible at these scales —
           * a single mid-scale pass keeps the signature lens bend at ~1/3 the
           * filter cost. Scale 14 (was 22/16/10) also calms the edge warp a notch. */}
          <feDisplacementMap in="src" in2="map" scale="14"
            xChannelSelector="R" yChannelSelector="G" result="chroma" />

          {/* Specular gloss — a soft moving wet highlight from a point light. */}
          <feGaussianBlur in="map" stdDeviation="6" result="bumpBlur" />
          <feSpecularLighting in="bumpBlur" surfaceScale="3"
            specularConstant="0.75" specularExponent="22"
            lightingColor="#ffffff" result="spec">
            <fePointLight x="120" y="-40" z="160" />
          </feSpecularLighting>
          <feComposite in="spec" in2="chroma" operator="in" result="specClip" />
          <feBlend in="chroma" in2="specClip" mode="screen" />
        </filter>
      </defs>
    </svg>
  );
}
