import { useRef, useEffect } from 'react';

export default function FullTourMap({ metadata, theme }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !metadata?.tour_positions) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const cssW = rect.width;
    const cssH = rect.height;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    // Background
    const bg = theme?.bg || [0.06, 0.06, 0.08];
    ctx.fillStyle = `rgb(${Math.round(bg[0]*255)},${Math.round(bg[1]*255)},${Math.round(bg[2]*255)})`;
    ctx.fillRect(0, 0, cssW, cssH);

    const positions = metadata.tour_positions;
    const tour = metadata.tour;
    if (!positions || positions.length === 0) return;

    // All city coords (not just tour order — we need original indices for labels)
    // tour_positions is in tour order, tour[i] is the city ID (1-indexed)

    // Compute bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of positions) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const pad = 50;
    const scale = Math.min((cssW - pad * 2) / rangeX, (cssH - pad * 2) / rangeY);
    const offX = (cssW - rangeX * scale) / 2;
    const offY = (cssH - rangeY * scale) / 2;

    const tx = (x) => offX + (x - minX) * scale;
    const ty = (y) => offY + (maxY - y) * scale;

    // Tour color
    const tourRGB = theme?.tourFound || [0.75, 0.22, 0.08];
    const tourColor = `rgba(${Math.round(tourRGB[0]*255)},${Math.round(tourRGB[1]*255)},${Math.round(tourRGB[2]*255)},0.7)`;

    // Draw tour path
    ctx.strokeStyle = tourColor;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let k = 0; k <= positions.length; k++) {
      const [px, py] = positions[k % positions.length];
      if (k === 0) ctx.moveTo(tx(px), ty(py));
      else ctx.lineTo(tx(px), ty(py));
    }
    ctx.stroke();

    // Draw edge direction arrows (subtle)
    ctx.fillStyle = tourColor;
    for (let k = 0; k < positions.length; k++) {
      const [x1, y1] = positions[k];
      const [x2, y2] = positions[(k + 1) % positions.length];
      const mx = (tx(x1) + tx(x2)) / 2;
      const my = (ty(y1) + ty(y2)) / 2;
      const dx = tx(x2) - tx(x1);
      const dy = ty(y2) - ty(y1);
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len < 20) continue; // skip very short edges
      const nx = dx / len, ny = dy / len;
      const sz = 4;
      ctx.beginPath();
      ctx.moveTo(mx + nx * sz, my + ny * sz);
      ctx.lineTo(mx - nx * sz - ny * sz * 0.6, my - ny * sz + nx * sz * 0.6);
      ctx.lineTo(mx - nx * sz + ny * sz * 0.6, my - ny * sz - nx * sz * 0.6);
      ctx.closePath();
      ctx.fill();
    }

    // Draw city dots
    const coreRGB = theme?.particleCore || [0.78, 0.24, 0.08];
    const dotColor = `rgb(${Math.round(coreRGB[0]*255)},${Math.round(coreRGB[1]*255)},${Math.round(coreRGB[2]*255)})`;
    const labelColor = theme?.label || 'rgba(200,195,185,0.7)';

    for (let i = 0; i < positions.length; i++) {
      const [px, py] = positions[i];
      const sx = tx(px), sy = ty(py);

      // Outer ring
      ctx.strokeStyle = dotColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.stroke();

      // Inner dot
      ctx.fillStyle = dotColor;
      ctx.beginPath();
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = labelColor;
      ctx.font = '11px JetBrains Mono, Consolas, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(String(tour[i]), sx, sy - 9);
    }

    // Stats overlay
    const statsY = cssH - 16;
    ctx.fillStyle = labelColor;
    ctx.font = '11px JetBrains Mono, Consolas, monospace';
    ctx.textAlign = 'left';
    const statsText = [];
    if (metadata.tour_distance != null)
      statsText.push(`Tour: ${Math.round(metadata.tour_distance).toLocaleString()}`);
    if (metadata.optimal_cost != null)
      statsText.push(`Optimal: ${Math.round(metadata.optimal_cost).toLocaleString()}`);
    if (metadata.gap_pct != null)
      statsText.push(`Gap: ${metadata.gap_pct.toFixed(2)}%`);
    ctx.fillText(statsText.join('  ·  '), pad, statsY);

  }, [metadata?.tour_positions, metadata?.tour, metadata?.tour_distance, theme]);

  return (
    <div className="w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
}
