import { PROTOCOL, RESULTS } from '../../data'

// The protocol, drawn: some IDs' normals train the detector; a held-out ID
// (never seen) is the whole test set, normals + anomalies both.
const TEST_ID = RESULTS[0]?.testId ?? '06'
const TRAIN_IDS = PROTOCOL.publicIds.filter((id) => id !== TEST_ID)
const sample = RESULTS[0]?.loio

export function LoioSplit() {
  return (
    <div className="loio">
      <div className="loio__side">
        <p className="loio__role">Train: normals only</p>
        <div className="loio__chips">
          {TRAIN_IDS.map((id, i) => (
            <span key={id} className="loio__chip loio__chip--train" style={{ animationDelay: `${i * 0.08}s` }}>
              id_{id}
            </span>
          ))}
        </div>
        <p className="loio__hint">healthy clips only, no anomalies</p>
      </div>

      <div className="loio__arrow" aria-hidden="true">
        →
      </div>

      <div className="loio__side">
        <p className="loio__role">Test: held-out ID</p>
        <div className="loio__chips">
          <span className="loio__chip loio__chip--test" style={{ animationDelay: '0.24s' }}>
            id_{TEST_ID}
          </span>
        </div>
        <p className="loio__hint">
          {sample ? `${sample.nNeg} normal + ${sample.nPos} anomaly (fan)` : 'normals + anomalies'}, a serial
          number training never saw
        </p>
      </div>
    </div>
  )
}
