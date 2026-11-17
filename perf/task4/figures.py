"""Dependency-free, portable SVG line figures."""
from html import escape

def svg(title,series,xlabel,ylabel):
 points=[p for ps in series.values() for p in ps];xm=max([p[0] for p in points] or [1]) or 1;ym=max([p[1] for p in points] or [1]) or 1
 x=lambda v:80+v/xm*660;y=lambda v:320-v/ym*240
 colors=['#2563eb','#dc2626','#059669','#9333ea','#d97706','#0891b2']
 s=['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="420" viewBox="0 0 880 420">','<rect width="880" height="420" fill="white"/>','<g font-family="sans-serif" font-size="13" fill="#111827">',f'<text x="25" y="26" font-size="18">{escape(title)}</text>','<path d="M80 65 V320 H755" stroke="#333" fill="none"/>']
 for i in range(6):
  v=ym*i/5;s.append(f'<text x="12" y="{y(v)+4:.1f}">{v:.3g}</text><path d="M80 {y(v):.1f} H750" stroke="#ddd"/>')
  v=xm*i/5;s.append(f'<text x="{x(v):.1f}" y="343">{v:.3g}</text>')
 for i,(label,ps) in enumerate(series.items()):
  color=colors[i%len(colors)];coords=' '.join(f'{x(a):.1f},{y(b):.1f}' for a,b in ps)
  s.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
  s.append(f'<text x="{90+(i%3)*250}" y="{385+(i//3)*18}" fill="{color}">{escape(label)}</text>')
 s.extend([f'<text x="330" y="365">{escape(xlabel)}</text>',f'<text x="82" y="56">{escape(ylabel)}</text>','</g></svg>'])
 return '\n'.join(s)
