"""Geometry kernel templates.

Each template is a complete ``__global__`` kernel that expects
``compute_force_mag`` and ``compute_wall_force_mag`` device functions to
already be defined (concatenated earlier by the kernel builder).

The builder assembles a compilation unit as::

    common + force_model + wall_model + geometry_template

and compiles it via ``cupy.RawKernel``.

All geometry kernels use a per-particle mass array (``const float* mass``)
for the integration step: ``v = (v + F*dt/m) * damping``.
"""

GEOMETRY_KERNELS = {

    # ==================================================================
    # PLANAR 2-D  (inner/outer concentric-circle walls)
    # ==================================================================
    "planar": r"""
extern "C" __global__
void nbody_step(float* pos, float* vel, const float* pairEq,
                const float* mass,
                float strength, float innerR, float outerR,
                float wallStrength, float wallRangeFrac,
                float dt, float damping, float maxSpeed,
                int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = pos[idx*2], py = pos[idx*2+1];
    float vx = vel[idx*2], vy = vel[idx*2+1];
    float fx = 0.0f, fy = 0.0f;
    float m = mass[idx];

    /* ---- pairwise forces ---- */
    for (int j = 0; j < N; j++) {
        if (j == idx) continue;

        float dx = px - pos[j*2];
        float dy = py - pos[j*2+1];
        float d  = sqrtf(dx*dx + dy*dy);
        if (d < 1e-10f) d = 1e-10f;

        float eq = pairEq[idx*N + j];
        float fmag = compute_force_mag(d, eq, strength);

        float inv_d = 1.0f / d;
        fx += fmag * dx * inv_d;
        fy += fmag * dy * inv_d;
    }

    /* ---- wall forces (concentric circles) ---- */
    float radius = sqrtf(px*px + py*py);
    float safeR  = fmaxf(radius, 1e-10f);
    float nx     = px / safeR;
    float ny     = py / safeR;

    float wallGap   = outerR - innerR;
    float wallRange = wallGap * wallRangeFrac;

    /* inner wall: dist_to_wall = radius - innerR */
    if (innerR > 0.001f) {
        float distInner = radius - innerR;
        float wf = compute_wall_force_mag(distInner, wallRange, wallStrength);
        /* positive wf pushes outward (away from inner wall) */
        fx += wf * nx;
        fy += wf * ny;
    }

    /* outer wall: dist_to_wall = outerR - radius */
    {
        float distOuter = outerR - radius;
        float wf = compute_wall_force_mag(distOuter, wallRange, wallStrength);
        /* positive wf pushes inward (away from outer wall) */
        fx -= wf * nx;
        fy -= wf * ny;
    }

    /* ---- integration (multiplicative damping) ---- */
    float dampFactor = fmaxf(0.01f, damping);
    vx = (vx + fx * dt / m) * dampFactor;
    vy = (vy + fy * dt / m) * dampFactor;

    float speed = sqrtf(vx*vx + vy*vy);
    if (speed > maxSpeed) {
        float s = maxSpeed / speed;
        vx *= s;
        vy *= s;
    }

    px += vx * dt;
    py += vy * dt;

    pos[idx*2]   = px;
    pos[idx*2+1] = py;
    vel[idx*2]   = vx;
    vel[idx*2+1] = vy;
}
""",

    # ==================================================================
    # TORUS 3-D  (tube wall around major circle)
    # ==================================================================
    "torus": r"""
extern "C" __global__
void nbody_step(float* pos, float* vel, const float* pairEq,
                const float* mass,
                float strength, float majorR, float tubeR,
                float wallStrength, float wallRangeFrac,
                float dt, float damping, float maxSpeed,
                int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = pos[idx*3], py = pos[idx*3+1], pz = pos[idx*3+2];
    float vx = vel[idx*3], vy = vel[idx*3+1], vz = vel[idx*3+2];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float m = mass[idx];

    /* ---- pairwise forces ---- */
    for (int j = 0; j < N; j++) {
        if (j == idx) continue;

        float dx = px - pos[j*3];
        float dy = py - pos[j*3+1];
        float dz = pz - pos[j*3+2];
        float d  = sqrtf(dx*dx + dy*dy + dz*dz);

        float eq = pairEq[idx*N + j];
        float fmag = compute_force_mag(d, eq, strength);

        float inv_d = 1.0f / fmaxf(d, 1e-10f);
        fx += fmag * dx * inv_d;
        fy += fmag * dy * inv_d;
        fz += fmag * dz * inv_d;
    }

    /* ---- tube wall force ---- */
    float theta = atan2f(py, px);
    float cx = majorR * cosf(theta);
    float cy = majorR * sinf(theta);
    float cz = 0.0f;

    float wx = px - cx, wy = py - cy, wz = pz - cz;
    float wdist = sqrtf(wx*wx + wy*wy + wz*wz);

    float wallDist  = tubeR - wdist;          /* distance to tube surface */
    float wallRange = tubeR * wallRangeFrac;

    if (wdist > 0.001f) {
        float wf = compute_wall_force_mag(wallDist, wallRange, wallStrength);
        /* Force directed inward toward tube centre (negative radial) */
        float inv_w = 1.0f / wdist;
        fx -= wf * wx * inv_w;
        fy -= wf * wy * inv_w;
        fz -= wf * wz * inv_w;
    }

    /* ---- integration with multiplicative damping ---- */
    vx = (vx + fx * dt / m) * damping;
    vy = (vy + fy * dt / m) * damping;
    vz = (vz + fz * dt / m) * damping;

    float speed = sqrtf(vx*vx + vy*vy + vz*vz);
    if (speed > maxSpeed) {
        float s = maxSpeed / speed;
        vx *= s; vy *= s; vz *= s;
    }

    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    /* ---- hard torus constraint ---- */
    theta = atan2f(py, px);
    cx = majorR * cosf(theta);
    cy = majorR * sinf(theta);
    wx = px - cx; wy = py - cy; wz = pz;
    wdist = sqrtf(wx*wx + wy*wy + wz*wz);

    if (wdist > tubeR * 0.99f) {
        float inv_w = 1.0f / fmaxf(wdist, 1e-8f);
        float nnx = wx * inv_w, nny = wy * inv_w, nnz = wz * inv_w;
        px = cx + nnx * tubeR * 0.98f;
        py = cy + nny * tubeR * 0.98f;
        pz = nnz * tubeR * 0.98f;
        float vn = vx*nnx + vy*nny + vz*nnz;
        vx = (vx - 2.0f*vn*nnx) * 0.7f;
        vy = (vy - 2.0f*vn*nny) * 0.7f;
        vz = (vz - 2.0f*vn*nnz) * 0.7f;
    }

    pos[idx*3]   = px; pos[idx*3+1] = py; pos[idx*3+2] = pz;
    vel[idx*3]   = vx; vel[idx*3+1] = vy; vel[idx*3+2] = vz;
}
""",

    # ==================================================================
    # TORUS CIRCLE  (1-D angular dynamics on S^1 — standalone kernel,
    #                does NOT use compute_force_mag / compute_wall_force_mag)
    # ==================================================================
    "torus_circle": r"""
extern "C" __global__
void circle_step(float* pos, float* vel, const float* pairSigma,
                 float ljStrength, float circleLjScale,
                 float majorR, float dt, float circleDamping,
                 float maxOmega, float sigmaArcScale,
                 int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = pos[idx*3], py = pos[idx*3+1];
    float vx = vel[idx*3], vy = vel[idx*3+1];
    float myTheta = atan2f(py, px);

    float force = 0.0f;

    for (int j = 0; j < N; j++) {
        if (j == idx) continue;

        float otherTheta = atan2f(pos[j*3+1], pos[j*3]);
        float dtheta = otherTheta - myTheta;

        /* wrap to (-pi, pi] */
        while (dtheta >  3.14159265f) dtheta -= 6.28318530f;
        while (dtheta < -3.14159265f) dtheta += 6.28318530f;

        float arcDist = fabsf(dtheta) * majorR;
        float sigma3d = pairSigma[idx*N + j];
        float sigmaArc = sigma3d * sigmaArcScale;
        float cutoff = sigmaArc * 3.0f;

        if (arcDist > 0.01f && arcDist < cutoff) {
            float r = fmaxf(arcDist, 0.01f);
            float sr = sigmaArc / r;
            float sr6 = sr * sr * sr * sr * sr * sr;
            float fmag = 24.0f * ljStrength * circleLjScale / r
                         * (2.0f * sr6 * sr6 - sr6);
            float sign = dtheta > 0.0f ? 1.0f : -1.0f;
            force += fmag * sign;
        }
    }

    /* angular integration */
    float tangentX = -sinf(myTheta), tangentY = cosf(myTheta);
    float omega = (vx * tangentX + vy * tangentY) / majorR;
    omega = (omega + force * dt / majorR) * circleDamping;
    if (fabsf(omega) > maxOmega) omega = copysignf(maxOmega, omega);

    myTheta += omega * dt;

    float cosT = cosf(myTheta), sinT = sinf(myTheta);
    pos[idx*3]   = majorR * cosT;
    pos[idx*3+1] = majorR * sinT;
    pos[idx*3+2] = 0.0f;
    vel[idx*3]   = -sinT * omega * majorR;
    vel[idx*3+1] =  cosT * omega * majorR;
    vel[idx*3+2] = 0.0f;
}
""",

    # ==================================================================
    # ANNULAR 2-D  (2-D projection of torus: ring between R-r and R+r)
    # ==================================================================
    "annular": r"""
extern "C" __global__
void nbody_step(float* pos, float* vel, const float* pairEq,
                const float* mass,
                float strength, float majorR, float tubeR,
                float wallStrength, float wallRangeFrac,
                float dt, float damping, float maxSpeed,
                int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = pos[idx*2], py = pos[idx*2+1];
    float vx = vel[idx*2], vy = vel[idx*2+1];
    float fx = 0.0f, fy = 0.0f;
    float m = mass[idx];

    /* ---- pairwise forces ---- */
    for (int j = 0; j < N; j++) {
        if (j == idx) continue;

        float dx = px - pos[j*2];
        float dy = py - pos[j*2+1];
        float d  = sqrtf(dx*dx + dy*dy);

        float eq = pairEq[idx*N + j];
        float fmag = compute_force_mag(d, eq, strength);

        float inv_d = 1.0f / fmaxf(d, 1e-10f);
        fx += fmag * dx * inv_d;
        fy += fmag * dy * inv_d;
    }

    /* ---- tube wall force (2-D: distance from major circle at radius R) ---- */
    float theta = atan2f(py, px);
    float cx = majorR * cosf(theta);
    float cy = majorR * sinf(theta);
    float wx = px - cx, wy = py - cy;
    float wdist = sqrtf(wx*wx + wy*wy);

    float wallDist  = tubeR - wdist;
    float wallRange = tubeR * wallRangeFrac;

    if (wdist > 0.001f) {
        float wf = compute_wall_force_mag(wallDist, wallRange, wallStrength);
        wf = fmaxf(-1000.0f, fminf(wf, 1000.0f));
        float inv_w = 1.0f / wdist;
        fx -= wf * wx * inv_w;
        fy -= wf * wy * inv_w;
    }

    /* ---- pre-integration hard constraint ---- */
    float radius = sqrtf(px*px + py*py);
    float distFromR = fabsf(radius - majorR);
    if (distFromR > tubeR * 0.99f && radius > 0.001f) {
        float targetR = majorR + copysignf(tubeR * 0.98f, radius - majorR);
        float inv_r = 1.0f / radius;
        px = px * inv_r * targetR;
        py = py * inv_r * targetR;
        float nnx = px * inv_r, nny = py * inv_r;
        float vn = vx*nnx + vy*nny;
        vx = (vx - 2.0f*vn*nnx) * 0.7f;
        vy = (vy - 2.0f*vn*nny) * 0.7f;
    }

    /* ---- integration (multiplicative damping) ---- */
    vx = (vx + fx * dt / m) * damping;
    vy = (vy + fy * dt / m) * damping;

    float speed = sqrtf(vx*vx + vy*vy);
    if (speed > maxSpeed) {
        float s = maxSpeed / speed;
        vx *= s; vy *= s;
    }

    px += vx * dt;
    py += vy * dt;

    /* ---- post-integration hard constraint ---- */
    radius = sqrtf(px*px + py*py);
    distFromR = fabsf(radius - majorR);
    if (distFromR > tubeR * 0.99f && radius > 0.001f) {
        float targetR = majorR + copysignf(tubeR * 0.98f, radius - majorR);
        float inv_r = 1.0f / radius;
        px = px * inv_r * targetR;
        py = py * inv_r * targetR;
        float nnx = px * inv_r, nny = py * inv_r;
        float vn = vx*nnx + vy*nny;
        vx = (vx - 2.0f*vn*nnx) * 0.7f;
        vy = (vy - 2.0f*vn*nny) * 0.7f;
    }

    pos[idx*2]   = px; pos[idx*2+1] = py;
    vel[idx*2]   = vx; vel[idx*2+1] = vy;
}
""",
}
