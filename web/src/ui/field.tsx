import { useEffect, useRef } from "react";

interface Hairline {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  phase: number;
  speed: number;
  weight: number;
}

const LINES: Hairline[] = [
  { ax: -0.08, ay: 0.22, bx: 0.78, by: 0.06, phase: 0.2, speed: 0.07, weight: 1 },
  { ax: 0.12, ay: 0.96, bx: 1.08, by: 0.52, phase: 1.5, speed: 0.05, weight: 0.7 },
  { ax: 0.58, ay: -0.06, bx: 0.98, by: 0.7, phase: 2.4, speed: 0.085, weight: 0.6 },
  { ax: -0.12, ay: 0.7, bx: 0.46, by: 0.4, phase: 3.2, speed: 0.055, weight: 1 },
  { ax: 0.32, ay: 0.08, bx: 0.58, by: 1.08, phase: 4.1, speed: 0.075, weight: 0.55 },
  { ax: 0.04, ay: 0.36, bx: 0.92, by: 0.82, phase: 5.2, speed: 0.042, weight: 0.75 },
];

export function Field() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame = 0;
    let time = 0;
    let last = performance.now();
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(window.innerWidth * dpr);
      canvas.height = Math.floor(window.innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (now: number) => {
      if (!reduced) frame = requestAnimationFrame(draw);
      if (document.hidden) return;
      const width = window.innerWidth;
      const height = window.innerHeight;
      const step = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (!reduced && document.visibilityState === "visible") time += step;
      ctx.clearRect(0, 0, width, height);
      ctx.lineCap = "round";
      for (const line of LINES) {
        const sway = Math.sin(time * line.speed + line.phase);
        const drift = Math.cos(time * line.speed * 0.67 + line.phase);
        const x1 = (line.ax + sway * 0.035) * width;
        const y1 = (line.ay + drift * 0.04) * height;
        const x2 = (line.bx + drift * 0.03) * width;
        const y2 = (line.by + sway * 0.028) * height;
        const ink = ctx.createLinearGradient(x1, y1, x2, y2);
        ink.addColorStop(0, "rgba(196, 107, 74, 0)");
        ink.addColorStop(0.28, "rgba(196, 107, 74, 0.11)");
        ink.addColorStop(0.62, "rgba(196, 107, 74, 0.07)");
        ink.addColorStop(1, "rgba(196, 107, 74, 0)");
        ctx.strokeStyle = ink;
        ctx.lineWidth = line.weight;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    };

    resize();
    frame = requestAnimationFrame(draw);
    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas ref={ref} className="field" aria-hidden="true" />;
}
