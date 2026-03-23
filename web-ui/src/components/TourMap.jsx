import { useRef, useEffect } from 'react';

export default function TourMap({ metadata, theme }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !metadata?.tour_positions) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssW, cssH);

    const positions = metadata.tour_positions;
    if (positions.length === 0) return;

    // Compute bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of positions) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const pad = 20;
    const scale = Math.min((cssW - pad * 2) / rangeX, (cssH - pad * 2) / rangeY);
    const offX = (cssW - rangeX * scale) / 2;
    const offY = (cssH - rangeY * scale) / 2;

    const tx = (x) => offX + (x - minX) * scale;
    const ty = (y) => offY + (maxY - y) * scale; // flip Y

    // Draw tour path
    const tourColor = theme?.tourFound
      ? `rgba(${Math.round(theme.tourFound[0]*255)},${Math.round(theme.tourFound[1]*255)},${Math.round(theme.tourFound[2]*255)},0.7)`
      : 'rgba(192,88,42,0.7)';

    ctx.strokeStyle = tourColor;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let k = 0; k <= positions.length; k++) {
      const [px, py] = positions[k % positions.length];
      if (k === 0) ctx.moveTo(tx(px), ty(py));
      else ctx.lineTo(tx(px), ty(py));
    }
    ctx.stroke();

    // Draw city dots
    const dotColor = theme?.particleCore
      ? `rgb(${Math.round(theme.particleCore[0]*255)},${Math.round(theme.particleCore[1]*255)},${Math.round(theme.particleCore[2]*255)})`
      : '#c0582a';

    for (let i = 0; i < positions.length; i++) {
      const [px, py] = positions[i];
      const sx = tx(px), sy = ty(py);

      // Dot
      ctx.fillStyle = dotColor;
      ctx.beginPath();
      ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = theme?.label || 'rgba(200,195,185,0.6)';
      ctx.font = '9px JetBrains Mono, Consolas, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(String(metadata.tour[i]), sx, sy - 6);
    }

  }, [metadata?.tour_positions, metadata?.tour, theme]);

  if (!metadata?.tour_positions) return null;

  return (
    <div className="border-t border-current/10 p-4">
      <h3 className="font-mono text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
        Tour Map
      </h3>
      <canvas
        ref={canvasRef}
        className="w-full rounded-md"
        style={{
          height: 220,
          background: theme?.bg
            ? `rgb(${Math.round(theme.bg[0]*255)},${Math.round(theme.bg[1]*255)},${Math.round(theme.bg[2]*255)})`
            : '#0f0f12',
          border: '1px solid rgba(255,255,255,0.06)',
        }}
      />
    </div>
  );
}
