import { useEffect, useRef, useState } from "react";
import { actionLabel, clearHeld, keyToHold, setHeld } from "./input/actions";
import { live, publish, subscribeLive } from "./live";
import { mountCrt, type CrtSurface } from "./render/crt";
import { requestReGround, startEngine, stopEngine } from "./runtime/engine";
import { useUi, type Mode } from "./store";
import { Field } from "./ui/field";

export function App() {
  const mode = useUi((state) => state.mode);
  const game = useUi((state) => state.game);
  const setMode = useUi((state) => state.setMode);
  const setGame = useUi((state) => state.setGame);

  useEffect(() => {
    void startEngine(mode);
    return () => stopEngine();
  }, [mode, game]);

  useEffect(() => {
    document.title = mode === "landing" ? "Dream Arcade" : `Dream Arcade — ${mode[0].toUpperCase()}${mode.slice(1)}`;
  }, [mode]);

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (controlHasFocus(event.target)) return;
      const hold = keyToHold(event.code);
      if (!hold) return;
      event.preventDefault();
      setHeld(hold, true);
    };
    const up = (event: KeyboardEvent) => {
      const hold = keyToHold(event.code);
      if (!hold) return;
      setHeld(hold, false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", clearHeld);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", clearHeld);
      clearHeld();
    };
  }, []);

  return (
    <div className="app">
      <Field />
      <Atmosphere />
      <header>
        <div className="games" role="group" aria-label="Game">
          <button type="button" aria-pressed={game === "pong"} className={game === "pong" ? "is-on" : ""} onClick={() => setGame("pong")}>Pong</button>
          <button type="button" aria-pressed={game === "breakout"} className={game === "breakout" ? "is-on" : ""} onClick={() => setGame("breakout")}>Breakout</button>
        </div>
        <button type="button" className="word" onClick={() => setMode("landing")}>Dream Arcade</button>
        <nav aria-label="Sections">
          <button type="button" aria-current={mode === "landing" ? "page" : undefined} className={mode === "landing" ? "is-on" : ""} onClick={() => setMode("landing")}>About</button>
          <button type="button" aria-current={mode === "play" ? "page" : undefined} className={mode === "play" ? "is-on" : ""} onClick={() => setMode("play")}>Play</button>
          <button type="button" aria-current={mode === "lab" ? "page" : undefined} className={mode === "lab" ? "is-on" : ""} onClick={() => setMode("lab")}>Lab</button>
          <button type="button" aria-current={mode === "agent" ? "page" : undefined} className={mode === "agent" ? "is-on" : ""} onClick={() => setMode("agent")}>Agent</button>
        </nav>
      </header>
      <main>
        {mode === "landing" ? <Landing game={game} onPlay={() => setMode("play")} /> : <Stage mode={mode} />}
      </main>
    </div>
  );
}

function controlHasFocus(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "BUTTON" || tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function Landing({ game, onPlay }: { game: "pong" | "breakout"; onPlay: () => void }) {
  const title = game === "breakout" ? "Breakout" : "Pong";
  return (
    <section className="landing">
      <div>
        <p className="kicker">A table, imagined</p>
        <h1>{title}, drawn<br /><em>from noise.</em></h1>
        <p className="deck">
          You move. A few Euler steps denoise the next frame from the last four.
          The picture is 64 by 64, and it drifts.
        </p>
        <button type="button" className="key" onClick={onPlay}>Play</button>
        <p className="fine">
          {game === "breakout"
            ? "Left and right move the paddle. Breakout weights come from dreamarcade smoke data=breakout, and they may still be a smoke checkpoint."
            : "Weights in this checkout may be a smoke checkpoint until a Kaggle run is exported."}
          {" "}Nothing on this page is a measured score.
        </p>
      </div>
      <figure className="poster">
        <div className="bezel">
          <p className="screen-name">The table</p>
          {game === "breakout" ? <BrickCourt /> : <Court />}
        </div>
        <figcaption>Original rules. Generated pixels.</figcaption>
      </figure>
    </section>
  );
}

function Atmosphere() {
  return (
    <div className="atmosphere" aria-hidden="true">
      <span className="tick tick-tl" />
      <span className="tick tick-tr" />
      <span className="tick tick-bl" />
      <span className="tick tick-br" />
      <p className="margin-note">fig. 01</p>
    </div>
  );
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(query.matches);
    const onChange = () => setReduced(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

function Court() {
  const ballRef = useRef<SVGCircleElement>(null);
  const leftRef = useRef<SVGRectElement>(null);
  const rightRef = useRef<SVGRectElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    if (reduced) return;
    let frame = 0;
    let last = performance.now();
    let x = 120;
    let y = 118;
    let vx = 78;
    let vy = 54;
    let leftY = 96;
    let rightY = 96;
    const height = 48;
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      if (document.hidden) return;
      const dt = Math.min(0.032, (now - last) / 1000);
      last = now;
      x += vx * dt;
      y += vy * dt;
      if (y < 22) { y = 22; vy = Math.abs(vy); }
      if (y > 218) { y = 218; vy = -Math.abs(vy); }
      leftY += (y - height / 2 - leftY) * Math.min(1, dt * 7);
      rightY += (y - height / 2 - rightY) * Math.min(1, dt * 7);
      if (x < 36 && vx < 0 && y > leftY && y < leftY + height) {
        x = 36;
        vx = Math.abs(vx);
        vy += (y - (leftY + height / 2)) * 1.4;
      }
      if (x > 204 && vx > 0 && y > rightY && y < rightY + height) {
        x = 204;
        vx = -Math.abs(vx);
        vy += (y - (rightY + height / 2)) * 1.4;
      }
      vy = Math.max(-110, Math.min(110, vy));
      if (x < 10 || x > 230) {
        x = 120;
        y = 118;
        vx = 78 * (x < 10 ? 1 : -1);
        vy = 48;
      }
      ballRef.current?.setAttribute("cx", x.toFixed(1));
      ballRef.current?.setAttribute("cy", y.toFixed(1));
      leftRef.current?.setAttribute("y", leftY.toFixed(1));
      rightRef.current?.setAttribute("y", rightY.toFixed(1));
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [reduced]);

  return (
    <svg className="court" viewBox="0 0 240 240" role="img" aria-label="A diagram of a Pong table">
      <rect width="240" height="240" fill="#070605" />
      <line x1="120" y1="16" x2="120" y2="224" stroke="#3a342c" strokeWidth="1.5" strokeDasharray="3 7" />
      <rect ref={leftRef} x="22" y="96" width="7" height="48" rx="1" fill="#f3eee6" />
      <rect ref={rightRef} x="211" y="96" width="7" height="48" rx="1" fill="#f3eee6" />
      <circle ref={ballRef} cx="120" cy="118" r="6" fill="#c46b4a" />
    </svg>
  );
}

function BrickCourt() {
  const rootRef = useRef<SVGSVGElement>(null);
  const reduced = useReducedMotion();
  const bricks = [];
  const colors = ["#b43c3c", "#b48c28", "#289650", "#325ab4"];
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 8; col++) {
      bricks.push(
        <rect key={`${row}-${col}`} data-brick="" x={12 + col * 27} y={18 + row * 12} width="22" height="8" fill={colors[row]} />,
      );
    }
  }

  useEffect(() => {
    if (reduced) return;
    const svg = rootRef.current;
    if (!svg) return;
    const ball = svg.querySelector("[data-ball]");
    const paddle = svg.querySelector("[data-paddle]");
    const bricks = [...svg.querySelectorAll("[data-brick]")];
    if (!(ball instanceof SVGCircleElement) || !(paddle instanceof SVGRectElement)) return;
    let frame = 0;
    let last = performance.now();
    let x = 120;
    let y = 150;
    let vx = 62;
    let vy = -88;
    let paddleX = 96;
    const serve = () => {
      x = 120;
      y = 168;
      vx = 54;
      vy = -90;
      for (const brick of bricks) brick.setAttribute("opacity", "1");
    };
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      if (document.hidden) return;
      const dt = Math.min(0.032, (now - last) / 1000);
      last = now;
      x += vx * dt;
      y += vy * dt;
      if (x < 16) { x = 16; vx = Math.abs(vx); }
      if (x > 224) { x = 224; vx = -Math.abs(vx); }
      if (y < 12) { y = 12; vy = Math.abs(vy); }
      paddleX += (x - 24 - paddleX) * Math.min(1, dt * 6);
      paddleX = Math.max(8, Math.min(184, paddleX));
      if (y > 190 && y < 206 && vy > 0 && x > paddleX && x < paddleX + 48) {
        y = 190;
        vy = -Math.abs(vy);
        vx += (x - (paddleX + 24)) * 1.6;
        vx = Math.max(-120, Math.min(120, vx));
      }
      for (const brick of bricks) {
        if (brick.getAttribute("opacity") === "0") continue;
        const bx = Number(brick.getAttribute("x"));
        const by = Number(brick.getAttribute("y"));
        if (x > bx - 4 && x < bx + 26 && y > by - 4 && y < by + 12) {
          brick.setAttribute("opacity", "0");
          vy = -vy;
          break;
        }
      }
      const gone = bricks.every((brick) => brick.getAttribute("opacity") === "0");
      if (y > 236 || gone) serve();
      ball.setAttribute("cx", x.toFixed(1));
      ball.setAttribute("cy", y.toFixed(1));
      paddle.setAttribute("x", paddleX.toFixed(1));
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [reduced]);

  return (
    <svg ref={rootRef} className="court" viewBox="0 0 240 240" role="img" aria-label="A diagram of a Breakout table">
      <rect width="240" height="240" fill="#070605" />
      {bricks}
      <rect data-paddle="" x="96" y="196" width="48" height="6" rx="1" fill="#f3eee6" />
      <circle data-ball="" cx="120" cy="168" r="5" fill="#c46b4a" />
    </svg>
  );
}

function Stage({ mode }: { mode: Exclude<Mode, "landing"> }) {
  const status = useLiveStatus();
  const knobs = useUi();
  const playRef = useRef<HTMLCanvasElement>(null);
  const dreamRef = useRef<HTMLCanvasElement>(null);
  const realRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const surfaces: CrtSurface[] = [];
    const mount = (canvas: HTMLCanvasElement | null, pick: () => Float32Array | null) => {
      if (!canvas) return;
      const crt = mountCrt(canvas);
      surfaces.push(crt);
      return { crt, pick };
    };
    const targets = [
      mount(playRef.current, () => live.playFrame),
      mount(dreamRef.current, () => live.dreamFrame),
    ];
    let realCrt: CrtSurface | null = null;
    if (realRef.current) {
      realCrt = mountCrt(realRef.current);
      surfaces.push(realCrt);
    }
    let frames = 0;
    let stamp = performance.now();
    let raf = 0;
    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (document.hidden) return;
      frames += 1;
      if (now - stamp > 500) {
        live.fps = (frames * 1000) / (now - stamp);
        frames = 0;
        stamp = now;
        publish();
      }
      for (const target of targets) {
        const frame = target?.pick();
        if (frame && target) target.crt.drawChw(frame);
      }
      if (realCrt && live.realFrame) realCrt.drawRgb(live.realFrame);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      for (const crt of surfaces) crt.destroy();
    };
  }, [mode]);

  if (status === "error") return <ErrorPanel />;

  const dreamName = mode === "lab" ? "Dream" : mode === "agent" ? "Agent" : "Dream";

  return (
    <section className={mode === "lab" ? "stage lab" : "stage"}>
      <div className="floor">
        <div className="screens">
          {mode === "lab" && (
            <figure className="bezel">
              <figcaption>Real</figcaption>
              <canvas ref={realRef} className="screen" width={512} height={512} aria-label="Recorded frame" />
            </figure>
          )}
          <figure className="bezel">
            <figcaption>{dreamName}</figcaption>
            <canvas ref={mode === "lab" ? dreamRef : playRef} className="screen" width={512} height={512} aria-label={mode === "agent" ? "Agent dream frame" : "Dream frame"} />
            {(mode === "play" || mode === "agent") && <TouchPad />}
          </figure>
        </div>
        {mode === "lab" && <DriftChart />}
        {mode === "agent" && (
          <p className="note">The actor was trained inside the dream, or it is still the smoke checkpoint.</p>
        )}
        {status === "loading" && <p className="loading" role="status">Loading the denoiser</p>}
      </div>
      <aside className="panel">
        <Stats mode={mode} />
        <p className="group-label">Sampler</p>
        <Slider label="Steps" min={1} max={5} step={1} value={knobs.nSteps} onChange={knobs.setNSteps} />
        {mode === "lab" && (
          <>
            <Slider label="History" min={1} max={4} step={1} value={knobs.historyLength} onChange={knobs.setHistoryLength} />
            <Slider label="Sample noise" min={0} max={1} step={0.01} value={knobs.churn} onChange={knobs.setChurn} />
            <Slider label="History noise" min={0} max={0.2} step={0.01} value={knobs.historyNoise} onChange={knobs.setHistoryNoise} />
            <label className="check">
              Free run
              <input type="checkbox" checked={knobs.freeRun} onChange={(event) => knobs.setFreeRun(event.target.checked)} />
            </label>
            <div className="actions">
              <button type="button" onClick={() => knobs.setPaused(!knobs.paused)}>{knobs.paused ? "Resume" : "Pause"}</button>
              <button type="button" onClick={requestReGround}>Re-ground</button>
            </div>
          </>
        )}
        {mode !== "lab" && (
          <div className="actions">
            <button type="button" onClick={requestReGround}>Re-ground</button>
          </div>
        )}
      </aside>
    </section>
  );
}

function ErrorPanel() {
  const snap = useLive();
  return (
    <section className="panel error" role="alert">
      <h2>{snap.errorTitle}</h2>
      <pre>{snap.errorBody}</pre>
    </section>
  );
}

function DriftChart() {
  const snap = useLive();
  return (
    <div className="drift">
      <span>Error along the tape</span>
      <Spark values={snap.maeSeries} />
    </div>
  );
}

function TouchPad() {
  const game = useUi((state) => state.game);
  if (game === "breakout") {
    return (
      <div className="touch">
        <button type="button" onPointerDown={() => setHeld("left", true)} onPointerUp={() => setHeld("left", false)} onPointerLeave={() => setHeld("left", false)} onPointerCancel={() => setHeld("left", false)}>Left</button>
        <button type="button" onPointerDown={() => setHeld("right", true)} onPointerUp={() => setHeld("right", false)} onPointerLeave={() => setHeld("right", false)} onPointerCancel={() => setHeld("right", false)}>Right</button>
      </div>
    );
  }
  return (
    <div className="touch">
      <button type="button" onPointerDown={() => setHeld("up", true)} onPointerUp={() => setHeld("up", false)} onPointerLeave={() => setHeld("up", false)} onPointerCancel={() => setHeld("up", false)}>Up</button>
      <button type="button" onPointerDown={() => setHeld("down", true)} onPointerUp={() => setHeld("down", false)} onPointerLeave={() => setHeld("down", false)} onPointerCancel={() => setHeld("down", false)}>Down</button>
    </div>
  );
}

function Row({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="row">
      <span>{label}</span>
      <span title={title}>{value}</span>
    </div>
  );
}

function Stats({ mode }: { mode: Exclude<Mode, "landing"> }) {
  const snap = useLive();
  const game = useUi((state) => state.game);
  const reward = snap.rewardLoaded && snap.predictedReward !== null ? snap.predictedReward.toFixed(2) : "not loaded";
  return (
    <div className="stats">
      <p className="group-label">Machine</p>
      <Row label="Provider" value={snap.provider ?? "…"} />
      <Row label="File" value={snap.modelFile ?? "…"} title={snap.modelFile ?? undefined} />
      <Row label="Render" value={`${snap.fps.toFixed(0)} fps`} />
      <Row label="Model" value={`${snap.ticksPerSec.toFixed(1)} ticks/s`} />
      <Row label="Tick" value={`${snap.lastTickMs.toFixed(0)} ms`} />
      <Row label="U-Net" value={`${snap.stepMs.toFixed(0)} ms`} />
      <p className="group-label">Table</p>
      <Row label="Action" value={actionLabel(snap.action, game)} />
      <Row label="Reward" value={reward} />
      <Row label="Score" value={snap.score.toFixed(2)} />
      <Row label="Done" value={snap.done === null ? "—" : snap.done ? "yes" : "no"} />
      {mode === "lab" && (
        <>
          <p className="group-label">Drift</p>
          <Row label="Frame" value={`${snap.replayIndex + 1}/${snap.replayFrames || "…"}`} />
          <Row label="Meter" value={snap.disagreement === null ? "…" : snap.disagreement.toFixed(3)} />
          <Row label="MAE" value={snap.maeSeries.at(-1)?.toFixed(3) ?? "…"} />
        </>
      )}
    </div>
  );
}

function Slider({ label, min, max, step, value, onChange }: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}) {
  const shown = Number.isInteger(step) ? String(value) : value.toFixed(2);
  return (
    <label className="slider">
      <span>{label}</span>
      <span>{shown}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function Spark({ values }: { values: number[] }) {
  const width = 240;
  const height = 56;
  const points = values.length === 0
    ? `0,${height - 4} ${width},${height - 4}`
    : values.map((value, index) => {
        const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width;
        const y = 6 + (1 - Math.min(1, Math.max(0, value))) * (height - 12);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="MAE drift">
      <line x1="0" y1={height - 4} x2={width} y2={height - 4} className="spark-base" />
      <polyline points={points} />
    </svg>
  );
}

function useLive() {
  const [version, setVersion] = useState(0);
  useEffect(() => subscribeLive(() => setVersion((item) => item + 1)), []);
  void version;
  return live;
}

function useLiveStatus() {
  const [status, setStatus] = useState(live.status);
  useEffect(() => subscribeLive(() => {
    setStatus((current) => (current === live.status ? current : live.status));
  }), []);
  return status;
}
