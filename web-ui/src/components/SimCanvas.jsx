import { useRef, useEffect } from 'react';

const TAU = Math.PI * 2;

// ---- Vector math ----
function len3(a) { return Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }

function genTorusWireVerts(R, r, sU = 48, sV = 24) {
  const verts = [];
  for (let i = 0; i <= sU; i++) {
    const u = (i / sU) * TAU, cu = Math.cos(u), su = Math.sin(u);
    for (let j = 0; j <= sV; j++) {
      const v = (j / sV) * TAU, cv = Math.cos(v), sv = Math.sin(v);
      verts.push((R + r * cv) * cu, (R + r * cv) * su, r * sv);
    }
  }
  const lines = [];
  for (let j = 0; j < sV; j++) for (let i = 0; i < sU; i++) {
    const a = i * (sV + 1) + j, b = (i + 1) * (sV + 1) + j;
    const ai = a * 3, bi = b * 3;
    lines.push(verts[ai], verts[ai+1], verts[ai+2], verts[bi], verts[bi+1], verts[bi+2]);
  }
  for (let i = 0; i <= sU; i++) for (let j = 0; j < sV; j++) {
    const a = i * (sV + 1) + j, b = a + 1;
    const ai = a * 3, bi = b * 3;
    lines.push(verts[ai], verts[ai+1], verts[ai+2], verts[bi], verts[bi+1], verts[bi+2]);
  }
  return new Float32Array(lines);
}

function genCircleVerts(R, segs = 128) {
  const v = [];
  for (let i = 0; i <= segs; i++) {
    const t = (i / segs) * TAU;
    v.push(R * Math.cos(t), R * Math.sin(t), 0);
  }
  return new Float32Array(v);
}

// ---- GL helpers ----
function compileShader(gl, src, type) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
  return s;
}
function linkProg(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compileShader(gl, vs, gl.VERTEX_SHADER));
  gl.attachShader(p, compileShader(gl, fs, gl.FRAGMENT_SHADER));
  gl.linkProgram(p);
  return p;
}

// ---- Matrix math ----
function perspective(fov, a, n, f) {
  const ff = 1/Math.tan(fov/2), nf = 1/(n-f);
  return [ff/a,0,0,0, 0,ff,0,0, 0,0,(f+n)*nf,-1, 0,0,2*f*n*nf,0];
}
function lookAt(e, c, u) {
  const sub = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const norm = (a) => { const l=Math.sqrt(dot(a,a)); return l>1e-8?[a[0]/l,a[1]/l,a[2]/l]:[0,0,0]; };
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const z=norm(sub(e,c)), x=norm(cross(u,z)), y=cross(z,x);
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1];
}
function mat4Mul(a,b) {
  const r=new Float32Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){r[j*4+i]=0;for(let k=0;k<4;k++)r[j*4+i]+=a[k*4+i]*b[j*4+k];}
  return r;
}
function ortho(l, r, b, t, n, f) {
  const lr=1/(r-l), bt=1/(t-b), nf=1/(n-f);
  return [2*lr,0,0,0, 0,2*bt,0,0, 0,0,2*nf,0, -(r+l)*lr,-(t+b)*bt,(f+n)*nf,1];
}

// ---- Shaders ----
const PARTICLE_VS = `
attribute vec3 aPos; attribute float aEnergy;
uniform mat4 uVP; uniform float uPointSize;
varying float vEnergy;
void main() {
  vEnergy = aEnergy;
  gl_Position = uVP * vec4(aPos, 1.0);
  gl_PointSize = uPointSize / max(gl_Position.w, 0.1);
}`;
const PARTICLE_FS = `
precision mediump float;
varying float vEnergy;
uniform vec3 uCore, uHot, uEdge;
void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float d = length(uv);
  if (d > 1.0) discard;
  float rim = smoothstep(0.35, 1.0, d);
  vec3 color = mix(mix(uCore, uHot, vEnergy), uEdge, rim);
  float alpha = smoothstep(1.0, 0.5, d);
  gl_FragColor = vec4(color, alpha);
}`;
const LINE_VS = `attribute vec3 aPos; uniform mat4 uVP; void main() { gl_Position = uVP * vec4(aPos, 1.0); }`;
const LINE_FS = `precision mediump float; uniform vec3 uColor; uniform float uAlpha; void main() { gl_FragColor = vec4(uColor, uAlpha); }`;

export default function SimCanvas({ stateRef, mode, theme, metadata }) {
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const progsRef = useRef(null);
  const bufsRef = useRef(null);
  const camRef = useRef({ angle: 0.4, pitch: 0.35, dist: 8.5, dragging: false, lx: 0, ly: 0 });
  const themeRef = useRef(theme);
  themeRef.current = theme;
  const metaRef = useRef(metadata);
  metaRef.current = metadata;

  // Init GL
  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
    if (!gl) return;
    glRef.current = gl;

    const particleProg = linkProg(gl, PARTICLE_VS, PARTICLE_FS);
    const lineProg = linkProg(gl, LINE_VS, LINE_FS);
    progsRef.current = { particleProg, lineProg };
    bufsRef.current = {
      posBuf: gl.createBuffer(), energyBuf: gl.createBuffer(), lineBuf: gl.createBuffer(),
    };

    // Resize
    const resize = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = devicePixelRatio;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      gl.viewport(0, 0, canvas.width, canvas.height);
    };
    resize();
    window.addEventListener('resize', resize);

    // Camera controls
    const cam = camRef.current;
    canvas.addEventListener('mousedown', e => { cam.dragging = true; cam.lx = e.clientX; cam.ly = e.clientY; });
    canvas.addEventListener('mousemove', e => {
      if (!cam.dragging) return;
      cam.angle += (e.clientX - cam.lx) * 0.005;
      cam.pitch = Math.max(-1.2, Math.min(1.2, cam.pitch + (e.clientY - cam.ly) * 0.005));
      cam.lx = e.clientX; cam.ly = e.clientY;
    });
    canvas.addEventListener('mouseup', () => cam.dragging = false);
    canvas.addEventListener('mouseleave', () => cam.dragging = false);
    canvas.addEventListener('wheel', e => { cam.dist = Math.max(3, Math.min(18, cam.dist + e.deltaY * 0.005)); e.preventDefault(); }, { passive: false });

    // Render loop
    let raf;
    const frame = () => {
      raf = requestAnimationFrame(frame);
      render(gl, progsRef.current, bufsRef.current, stateRef.current, camRef.current, mode, themeRef.current, metaRef.current);
    };
    raf = requestAnimationFrame(frame);

    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
  }, [stateRef, mode]);

  return <canvas ref={canvasRef} className="w-full h-full block" />;
}

function render(gl, progs, bufs, state, cam, mode, theme, meta) {
  if (!state.positions || state.N === 0) {
    gl.clearColor(theme.bg[0], theme.bg[1], theme.bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return;
  }

  const aspect = gl.canvas.width / gl.canvas.height;

  if (mode === 'torus') {
    renderTorus(gl, progs, bufs, state, cam, aspect, theme, meta);
  } else {
    // Both '2d' and 'annular' use the 2D renderer
    render2D(gl, progs, bufs, state, aspect, theme, meta);
  }
}

function renderTorus(gl, progs, bufs, state, cam, aspect, theme, meta) {
  const { N, R, r, r0, phase, positions, velocities } = state;
  const circlePhase = phase === 2;

  if (!cam.dragging) cam.angle += 0.0015;

  const eye = [
    cam.dist * Math.cos(cam.pitch) * Math.cos(cam.angle),
    cam.dist * Math.cos(cam.pitch) * Math.sin(cam.angle),
    cam.dist * Math.sin(cam.pitch),
  ];
  const vp = mat4Mul(perspective(0.8, aspect, 0.1, 100), lookAt(eye, [0,0,0], [0,0,1]));

  gl.clearColor(theme.bg[0], theme.bg[1], theme.bg[2], 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  drawParticles(gl, progs, bufs, positions, velocities, N, 3, vp,
    circlePhase ? 90 : 60, theme.particleCore, theme.particleHot, theme.particleEdge);

  if (!circlePhase && r0 > 0.01) {
    const wireVerts = genTorusWireVerts(R, r);
    const alpha = Math.max(0.04, (r / r0) * 0.28);
    drawLines(gl, progs.lineProg, bufs.lineBuf, wireVerts, vp, theme.torus, alpha);
  }

  if (circlePhase) {
    const circVerts = genCircleVerts(R);
    drawLineStrip(gl, progs.lineProg, bufs.lineBuf, circVerts, vp, theme.circle, 0.15);

    // Draw tour path on the circle
    if (meta?.tour) {
      const tourVerts = buildTourVerts3D(positions, N, meta.tour, R, -0.12);
      if (tourVerts) drawLineStrip(gl, progs.lineProg, bufs.lineBuf, tourVerts, vp, theme.tourFound, 0.65);
    }
  }
}

function render2D(gl, progs, bufs, state, aspect, theme, meta) {
  const { N, innerR, outerR, positions, velocities } = state;
  const norm = outerR > 0 ? outerR : 1;
  const pad = 0.15, viewLimit = 1.0 + pad;
  const vp = ortho(-viewLimit * aspect, viewLimit * aspect, -viewLimit, viewLimit, -1, 1);

  gl.clearColor(theme.bg[0], theme.bg[1], theme.bg[2], 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  drawCircle(gl, progs.lineProg, bufs.lineBuf, innerR / norm, vp, theme.wallInner, 0.8);
  drawCircle(gl, progs.lineProg, bufs.lineBuf, 1.0, vp, theme.wallOuter, 0.8);

  // Major circle for annular mode (midpoint between walls)
  if (state.mode === 'annular' && innerR > 0.01) {
    const majorR = (innerR + outerR) / 2 / norm;
    drawCircle(gl, progs.lineProg, bufs.lineBuf, majorR, vp, theme.circle || [0.4, 0.4, 0.5], 0.2);
  }

  // Tour path (normalized to outer radius)
  if (meta?.tour_positions && state.phase === 2) {
    const tourVerts = buildTourVerts2D(meta.tour_positions, norm);
    drawLineStrip(gl, progs.lineProg, bufs.lineBuf, tourVerts, vp, theme.tourFound || [0.0, 0.8, 0.2], 0.7);
  }

  const pos3 = new Float32Array(N * 3);
  const vel3 = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    pos3[i*3] = positions[i*2] / norm;
    pos3[i*3+1] = positions[i*2+1] / norm;
    pos3[i*3+2] = 0;
    vel3[i*3] = velocities[i*2];
    vel3[i*3+1] = velocities[i*2+1];
    vel3[i*3+2] = 0;
  }
  drawParticles(gl, progs, bufs, pos3, vel3, N, 3, vp, 10,
    theme.particleCore, theme.particleHot, theme.particleEdge);
}

// ---- Tour path helpers ----
function buildTourVerts3D(positions, N, tour, R, offset) {
  // Build vertices for tour path on the major circle (torus mode)
  // tour is 1-indexed city IDs
  if (!tour || tour.length === 0) return null;
  const verts = new Float32Array((tour.length + 1) * 3);
  for (let k = 0; k <= tour.length; k++) {
    const cityIdx = tour[k % tour.length] - 1; // 0-indexed
    if (cityIdx < 0 || cityIdx >= N) continue;
    const theta = Math.atan2(positions[cityIdx * 3 + 1], positions[cityIdx * 3]);
    const rr = R + offset;
    verts[k * 3] = rr * Math.cos(theta);
    verts[k * 3 + 1] = rr * Math.sin(theta);
    verts[k * 3 + 2] = 0;
  }
  return verts;
}

function buildTourVerts2D(tourPositions, norm) {
  // Build vertices for tour path in 2D (original coords, normalized)
  const n = tourPositions.length;
  const verts = new Float32Array((n + 1) * 3);
  for (let k = 0; k <= n; k++) {
    const pos = tourPositions[k % n];
    verts[k * 3] = pos[0] / norm;
    verts[k * 3 + 1] = pos[1] / norm;
    verts[k * 3 + 2] = 0;
  }
  return verts;
}

// ---- Drawing helpers ----
function drawParticles(gl, progs, bufs, pos, vel, N, stride, vp, ptSize, core, hot, edge) {
  const { particleProg } = progs;
  const { posBuf, energyBuf } = bufs;

  const energyData = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const vx = vel[i*stride], vy = vel[i*stride+1], vz = stride === 3 ? vel[i*stride+2] : 0;
    energyData[i] = Math.min(1, Math.sqrt(vx*vx+vy*vy+vz*vz) * 2.5);
  }

  gl.useProgram(particleProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(particleProg, 'uVP'), false, new Float32Array(vp));
  gl.uniform1f(gl.getUniformLocation(particleProg, 'uPointSize'), ptSize);
  gl.uniform3fv(gl.getUniformLocation(particleProg, 'uCore'), core);
  gl.uniform3fv(gl.getUniformLocation(particleProg, 'uHot'), hot);
  gl.uniform3fv(gl.getUniformLocation(particleProg, 'uEdge'), edge);

  const aPos = gl.getAttribLocation(particleProg, 'aPos');
  const aEnergy = gl.getAttribLocation(particleProg, 'aEnergy');

  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, energyBuf);
  gl.bufferData(gl.ARRAY_BUFFER, energyData, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aEnergy);
  gl.vertexAttribPointer(aEnergy, 1, gl.FLOAT, false, 0, 0);

  gl.drawArrays(gl.POINTS, 0, N);
  gl.disableVertexAttribArray(aPos);
  gl.disableVertexAttribArray(aEnergy);
}

function drawLines(gl, prog, buf, verts, vp, color, alpha) {
  gl.useProgram(prog);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uVP'), false, new Float32Array(vp));
  gl.uniform3fv(gl.getUniformLocation(prog, 'uColor'), color);
  gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), alpha);
  const aPos = gl.getAttribLocation(prog, 'aPos');
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.LINES, 0, verts.length / 3);
  gl.disableVertexAttribArray(aPos);
}

function drawLineStrip(gl, prog, buf, verts, vp, color, alpha) {
  gl.useProgram(prog);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uVP'), false, new Float32Array(vp));
  gl.uniform3fv(gl.getUniformLocation(prog, 'uColor'), color);
  gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), alpha);
  const aPos = gl.getAttribLocation(prog, 'aPos');
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.LINE_STRIP, 0, verts.length / 3);
  gl.disableVertexAttribArray(aPos);
}

function drawCircle(gl, prog, buf, radius, vp, color, alpha) {
  if (radius < 0.001) return;
  const verts = genCircleVerts(radius, 100);
  drawLineStrip(gl, prog, buf, verts, vp, color, alpha);
}
