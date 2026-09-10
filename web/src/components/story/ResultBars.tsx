import { RESULTS } from '../../data'

// One group per machine type: the raw baseline (red) vs the shipped pipeline
// (green) AUC-ROC, against a dashed 0.5 chance line. The jump is the story.
const W = 460
const PAD_L = 70
const PAD_R = 24
const TOP = 26
const GROUP_H = 52
const TRACK = W - PAD_L - PAD_R
const x = (v: number) => PAD_L + Math.max(0, v) * TRACK

export function ResultBars() {
  const H = TOP + RESULTS.length * GROUP_H + 22
  return (
    <svg
      className="chart-svg"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={
        'AUC-ROC per machine type, raw baseline versus the shipped pipeline, held out by model ID. ' +
        RESULTS.map((r) => `${r.type}: baseline ${r.baseline?.aucRoc ?? 'n/a'}, shipped ${r.loio.aucRoc}`).join('; ') +
        '. The dashed line is 0.5 chance.'
      }
    >
      <line x1={x(0.5)} y1={TOP - 8} x2={x(0.5)} y2={H - 20} stroke="var(--line-strong)" strokeDasharray="4 4" />
      <text x={x(0.5)} y={TOP - 12} fill="var(--fg-low)" fontSize="11" textAnchor="middle" fontFamily="var(--font-mono)">
        chance 0.5
      </text>

      {RESULTS.map((r, i) => {
        const yTop = TOP + i * GROUP_H
        const base = r.baseline?.aucRoc ?? 0
        return (
          <g key={r.type}>
            <text x={PAD_L - 8} y={yTop + 22} fill="var(--fg)" fontSize="13" textAnchor="end">
              {r.type}
            </text>
            <rect
              className="bar"
              style={{ animationDelay: `${i * 0.1}s` }}
              x={PAD_L}
              y={yTop + 2}
              width={base * TRACK}
              height={14}
              rx="2"
              fill="var(--bad)"
            />
            <text x={x(base) + 6} y={yTop + 13} fill="var(--fg-hi)" fontSize="11" fontFamily="var(--font-mono)">
              {base.toFixed(3)}
            </text>
            <rect
              className="bar"
              style={{ animationDelay: `${i * 0.1 + 0.05}s` }}
              x={PAD_L}
              y={yTop + 20}
              width={r.loio.aucRoc * TRACK}
              height={14}
              rx="2"
              fill="var(--good)"
            />
            <text x={x(r.loio.aucRoc) + 6} y={yTop + 31} fill="var(--fg-hi)" fontSize="11" fontFamily="var(--font-mono)">
              {r.loio.aucRoc.toFixed(3)}
            </text>
          </g>
        )
      })}

      {[0, 0.5, 1].map((t) => (
        <text key={t} x={x(t)} y={H - 6} fill="var(--fg-low)" fontSize="10" textAnchor="middle" fontFamily="var(--font-mono)">
          {t}
        </text>
      ))}
      <text x={PAD_L} y={H - 6} fill="var(--bad)" fontSize="10" textAnchor="start" fontFamily="var(--font-mono)">
        ■ baseline
      </text>
      <text x={PAD_L + 62} y={H - 6} fill="var(--good)" fontSize="10" textAnchor="start" fontFamily="var(--font-mono)">
        ■ shipped
      </text>
    </svg>
  )
}
