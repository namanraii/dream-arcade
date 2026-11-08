import { FRAME } from "../sampler/edm";
import { chwToRgba, rgbToRgba } from "./pixels";

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const FRAG = `
precision mediump float;
uniform sampler2D u_tex;
varying vec2 v_uv;
void main() {
  vec3 color = texture2D(u_tex, v_uv).rgb;
  float scan = fract(gl_FragCoord.y * 0.5);
  color *= scan < 0.5 ? 0.92 : 1.0;
  float vignette = smoothstep(1.15, 0.42, length(v_uv - 0.5));
  color *= mix(0.86, 1.0, vignette);
  color *= vec3(1.03, 0.98, 0.93);
  gl_FragColor = vec4(color, 1.0);
}
`;

export interface RgbView {
  data: Uint8Array;
  width: number;
  height: number;
  offset?: number;
}

export interface CrtSurface {
  drawChw(frame: Float32Array): void;
  drawRgb(frame: RgbView): void;
  destroy(): void;
  readonly webgl: boolean;
}

export function mountCrt(canvas: HTMLCanvasElement): CrtSurface {
  const gl = canvas.getContext("webgl", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) return mount2d(canvas);
  const program = link(gl, VERT, FRAG);
  if (!program) {
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return mount2d(canvas);
  }
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const position = gl.getAttribLocation(program, "a_pos");
  gl.useProgram(program);
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  let width = 0;
  let height = 0;

  const upload = (pixels: Uint8ClampedArray, w: number, h: number) => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const viewW = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const viewH = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== viewW || canvas.height !== viewH) {
      canvas.width = viewW;
      canvas.height = viewH;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    if (w !== width || h !== height) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      width = w;
      height = h;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  };

  return {
    webgl: true,
    drawChw(frame) {
      upload(chwToRgba(frame), FRAME, FRAME);
    },
    drawRgb(frame) {
      upload(rgbToRgba(frame.data, frame.offset ?? 0, frame.width, frame.height), frame.width, frame.height);
    },
    destroy() {
      gl.deleteTexture(texture);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
    },
  };
}

function mount2d(canvas: HTMLCanvasElement): CrtSurface {
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) {
    throw new Error("neither WebGL nor canvas 2D is available");
  }
  const scratch = document.createElement("canvas");
  const scratchCtx = scratch.getContext("2d");
  if (!scratchCtx) throw new Error("canvas 2D is not available");

  const paint = (pixels: Uint8ClampedArray, w: number, h: number) => {
    if (scratch.width !== w || scratch.height !== h) {
      scratch.width = w;
      scratch.height = h;
    }
    const owned = new Uint8ClampedArray(pixels);
    scratchCtx.putImageData(new ImageData(owned, w, h), 0, 0);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const viewW = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const viewH = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== viewW || canvas.height !== viewH) {
      canvas.width = viewW;
      canvas.height = viewH;
    }
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(scratch, 0, 0, canvas.width, canvas.height);
  };

  return {
    webgl: false,
    drawChw(frame) {
      paint(chwToRgba(frame), FRAME, FRAME);
    },
    drawRgb(frame) {
      paint(rgbToRgba(frame.data, frame.offset ?? 0, frame.width, frame.height), frame.width, frame.height);
    },
    destroy() {},
  };
}

function link(gl: WebGLRenderingContext, vert: string, frag: string): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;
  const vs = shader(gl, gl.VERTEX_SHADER, vert);
  const fs = shader(gl, gl.FRAGMENT_SHADER, frag);
  if (!vs || !fs) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.bindAttribLocation(program, 0, "a_pos");
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
  return program;
}

function shader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const item = gl.createShader(type);
  if (!item) return null;
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) {
    gl.deleteShader(item);
    return null;
  }
  return item;
}
