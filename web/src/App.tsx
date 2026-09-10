import { LoioSplit } from './components/story/LoioSplit'
import { ResultBars } from './components/story/ResultBars'
import { StoryBeat } from './components/story/StoryBeat'
import { BASELINE, DATASET, PENDING_TYPES, PROTOCOL, RESULTS } from './data'

const TOC = [
  { href: '#problem', label: 'The problem' },
  { href: '#solution', label: 'The fix' },
  { href: '#protocol', label: 'The protocol' },
  { href: '#architecture', label: 'How it fits together' },
  { href: '#usage', label: 'Running it yourself' },
  { href: '#next', label: 'Where it goes next' },
]

const STACK = [
  { name: 'stdlib wave', role: 'Reads the audio', how: BASELINE.audio },
  { name: 'scipy + numpy', role: 'Turns sound into the fingerprint', how: BASELINE.features },
  { name: 'scikit-learn', role: 'Normalizes per machine, then scores', how: BASELINE.detector },
]

const nTypes = RESULTS.length
const testId = RESULTS[0]?.testId ?? '06'
const trainIds = PROTOCOL.publicIds.filter((id) => id !== testId)
const farPct = Math.round(PROTOCOL.farTarget * 100)
const mean = (f: (r: (typeof RESULTS)[number]) => number) => RESULTS.reduce((s, r) => s + f(r), 0) / nTypes
const meanBefore = mean((r) => r.baseline?.aucRoc ?? 0)
const meanAfter = mean((r) => r.loio.aucRoc)
const cleared = RESULTS.filter((r) => r.loio.aucRoc >= 0.9).length
const byAuc = [...RESULTS].sort((a, b) => a.loio.aucRoc - b.loio.aucRoc)
const worst = byAuc[0]

function verdict(auc: number): { text: string; color: string } {
  if (auc >= 0.9) return { text: 'strong', color: 'var(--good)' }
  if (auc >= 0.8) return { text: 'usable', color: 'var(--amber)' }
  if (auc >= 0.5) return { text: 'weak', color: 'var(--amber)' }
  return { text: 'below chance', color: 'var(--bad)' }
}

export function App() {
  return (
    <>
      <a className="skip-link" href="#problem">
        Skip to the walkthrough
      </a>

      <div className="masthead">
        <div className="masthead__inner">
          <div className="masthead__mark">
            <b>Machine Debugger</b> · a walkthrough
          </div>
          <ul className="masthead__nav">
            <li>
              <a href="#problem">problem</a>
            </li>
            <li>
              <a href="#solution">the fix</a>
            </li>
            <li>
              <a href="#architecture">how it fits</a>
            </li>
            <li>
              <a href="#usage">run it</a>
            </li>
          </ul>
        </div>
      </div>

      <main className="page">
        <header className="page-hero">
          <p className="meta">Walkthrough · unsupervised machine-sound anomaly detection</p>
          <h1>Machine Debugger</h1>
          <p className="lead">
            Factories record the hum of a running machine and want one thing back: a warning before it
            breaks. I trained a detector only on the sound of healthy machines, then asked the question
            that matters on a factory floor. Does &quot;this one sounds broken&quot; still hold up on a
            machine model the detector has never heard?
          </p>
          <p className="intro-detail">
            My first cut failed badly. On {DATASET.channels}-channel MIMII recordings ({DATASET.sr / 1000} kHz,{' '}
            {PROTOCOL.publicIds.length} model IDs), scored on a held-out model ID, it averaged{' '}
            {meanBefore.toFixed(2)} AUC, no better than a coin flip, and it was worse than chance on fan and
            valve. The features were fine. What was missing was any handling of the machine-to-machine shift.
            Once I centered each machine on its own sound, added two features for transient faults, and swapped
            in a stronger detector, the average went to {meanAfter.toFixed(2)}, with {cleared} of {nTypes} types
            above 0.92. This walks through the failure, the fix, and where it still falls short.
          </p>
          <nav aria-label="On this page">
            <ul className="toc">
              {TOC.map((item) => (
                <li key={item.href}>
                  <a href={item.href}>{item.label}</a>
                </li>
              ))}
            </ul>
          </nav>
        </header>

        <StoryBeat
          id="problem"
          kicker="The problem"
          title="A new serial number is a new machine"
          caption="Four model IDs per type in MIMII. They are the same product, built and mixed on different lines, so their healthy sound already differs before anything breaks."
          visual={<LoioSplit />}
        >
          <p>
            An anomaly detector for machine sound is an easy thing to fool yourself with. Train it on a
            machine, test it on the same machine, and it looks sharp. But the moment you point it at a fresh
            unit off the line, the ordinary differences between two serial numbers can read as
            &quot;broken&quot; all on their own, before anything has gone wrong.
          </p>
          <p>
            MIMII makes this harder on purpose. Real factory background noise is mixed over every clip, so the
            detector has to separate a fault from both a new machine&apos;s normal voice and the room around
            it. That is the job I wanted to measure. The same-machine version only flatters the result.
          </p>
          <p>
            So the question I set is narrow: train on healthy clips from some model IDs, then score a
            different ID the detector never heard. Does &quot;sounds broken&quot; carry across the serial
            number, or does the detector just learn one specific machine?
          </p>
        </StoryBeat>

        <StoryBeat
          id="solution"
          kicker="The fix"
          title="Center each machine on its own sound, then flag the odd clips"
          caption="Red is the first-cut baseline, green is the shipped pipeline, both scored on a held-out model ID. The jump comes almost entirely from normalizing each machine before comparing."
          visual={<ResultBars />}
        >
          <p>
            The pipeline turns each ten-second clip into a small fingerprint: a log-mel spectrogram, which is
            a chart of how much energy sits in each pitch band as the clip plays, summarized over time by its
            average, spread, 95th percentile, and frame-to-frame change. The percentile and the change term
            keep the brief, spiky faults that a plain average would wash out. Then a detector learns what
            healthy fingerprints look like from the training machines and scores anything that sits far from
            them.
          </p>
          <p>
            The first version of this scored {meanBefore.toFixed(2)} on average, below a coin flip. The one
            change that fixed it: before comparing anything, I center each machine on its own statistics, so a
            fresh unit and a training unit are lined up and only a real fault stands out. On top of that I
            score with a Local Outlier Factor and a Gaussian mixture together rather than the plainer detector
            I started with. Same clips, same split, same healthy-only training.
          </p>
          <p>
            The chart is one pair of bars per machine type. Red is where I started, green is where it landed,
            both on a machine the detector never trained on. Every red bar is near or below the 0.5 chance
            line; every green bar clears it, and three of four clear 0.9.
          </p>

          <table className="choice-table">
            <caption className="sr-only">Baseline versus shipped AUC and catch rate per machine type, held-out ID {testId}, 0 dB</caption>
            <thead>
              <tr>
                <th scope="col">Type</th>
                <th scope="col">AUC before</th>
                <th scope="col">AUC after</th>
                <th scope="col">catch before</th>
                <th scope="col">catch after</th>
              </tr>
            </thead>
            <tbody>
              {RESULTS.map((r) => {
                const b = r.baseline
                const v = verdict(r.loio.aucRoc)
                return (
                  <tr key={r.type}>
                    <td>{r.type}</td>
                    <td style={{ color: 'var(--bad)', fontFamily: 'var(--font-mono)' }}>{(b?.aucRoc ?? 0).toFixed(3)}</td>
                    <td style={{ color: v.color, fontFamily: 'var(--font-mono)' }}>{r.loio.aucRoc.toFixed(3)}</td>
                    <td style={{ color: 'var(--fg-low)', fontFamily: 'var(--font-mono)' }}>{Math.round((b?.tprAtFar ?? 0) * 100)}%</td>
                    <td style={{ color: v.color, fontFamily: 'var(--font-mono)' }}>{Math.round(r.loio.tprAtFar * 100)}%</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="meta" style={{ textTransform: 'none', letterSpacing: 0 }}>
            Catch is the share of faulty clips flagged at a {farPct}% false-alarm rate on healthy ones.
          </p>

          <h3>What the pipeline concludes, and why I trust it</h3>
          <p>
            The result I trust is the ranking, and I trust it because no test machine ever enters training, so
            none of these numbers can be inflated by the detector memorizing one unit. The conclusion is that
            the machine-to-machine shift, not the sound of the faults, was doing most of the damage. Once that
            is removed, a plain fingerprint separates healthy from faulty well on fan, pump, and slider, and
            only partway on {worst.type}.
          </p>
          <p>
            One caveat I will not bury: to center a machine on its own sound, the detector uses a batch of that
            machine&apos;s own unlabeled clips. It never sees fault labels, and it never trains on the held-out
            ID, so this is not leakage. It is a normalization step, and it matches how you would deploy this,
            by letting the sensor record a new machine for a while before it starts judging it.
          </p>
          <p>
            {worst.type} is the one that stays short, at {worst.loio.aucRoc.toFixed(2)} AUC. Its normal sound is
            already a train of clicks, so a fault is a change in the pattern rather than a single odd moment. I
            tried three ways to catch that pattern: scoring frame by frame, a modulation feature for the click
            rhythm, and a small CNN embedding trained to tell the machines apart. None of them beat the plain
            summary, and two made it worse. The CNN shows why: it reached 100% training accuracy and just
            memorized the four units, because four machine IDs is not enough for a learned embedding to
            generalize a fault from. So {worst.type}&apos;s ceiling here looks like a data limit rather than an
            algorithm I have not tried. It needs more machine IDs, or the quieter 6 dB recordings, to climb.
          </p>

          {PENDING_TYPES.length > 0 ? (
            <div className="pending-note">
              {PENDING_TYPES.join(', ')} are not measured yet. Run <code>python scripts/run_all.py</code> to add
              them.
            </div>
          ) : null}
        </StoryBeat>

        <StoryBeat
          id="protocol"
          kicker="The protocol"
          title="Why the held-out test is the only number I report"
          caption="Three IDs' healthy clips train the detector; the fourth ID, which it never saw, is the entire test set, normals and faults both."
          visual={<LoioSplit />}
        >
          <p>
            The result rests on one rule, so let me be precise about it. Every number I report is{' '}
            <strong>leave-one-model-ID-out</strong>: the detector trains on healthy clips from IDs{' '}
            {trainIds.join(', ')} and is scored only on ID {testId}, which it never heard. I also run the
            same-unit version, but I keep it as a debug check and never put it on the headline, because testing
            on a machine you trained on leaks the answer.
          </p>
          <p>
            Training uses healthy clips only. The detector never sees a labelled fault while it learns; it just
            learns what normal sounds like and flags whatever sits far from that. Faults only appear at test
            time. A guard in the code hard-fails if a fault ever lands in training or if the held-out ID leaks
            into it, and a small test injects that leak on purpose to prove the guard fires. That guard is why I
            can stand behind the numbers.
          </p>
          <p>
            I report AUC and PR-AUC under that shift, plus the catch rate at a fixed false-alarm rate. I do not
            report plain accuracy: the test set is mostly healthy, so a detector that flags nothing would score
            in the nineties and warn no one.
          </p>
        </StoryBeat>

        <StoryBeat
          id="architecture"
          kicker="How it fits together"
          title="The pieces, then the choices I made"
          caption="First cut on the left, what I shipped on the right."
          visual={
            <div className="teach-card">
              <h3 className="teach-card__title">First cut vs. what I shipped</h3>
              <table className="choice-table">
                <caption className="sr-only">Design decisions</caption>
                <thead>
                  <tr>
                    <th scope="col">First cut</th>
                    <th scope="col">What I shipped</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Compare every machine on the same scale</td>
                    <td>Center each machine on its own sound first</td>
                  </tr>
                  <tr>
                    <td>Mean and spread of the spectrogram</td>
                    <td>Add the 95th percentile and frame-to-frame change</td>
                  </tr>
                  <tr>
                    <td>IsolationForest</td>
                    <td>Local Outlier Factor and a Gaussian mixture, combined</td>
                  </tr>
                  <tr>
                    <td>Report clip accuracy</td>
                    <td>Report AUC and catch rate at a fixed false-alarm rate</td>
                  </tr>
                </tbody>
              </table>
            </div>
          }
        >
          <p>
            Here is how the whole thing is put together, and then the calls I would defend. It is five small
            steps, each in its own file, with one script that runs them in order. <code>io.py</code> opens the
            raw audio files and pulls out a single channel. <code>features.py</code> turns a clip into the
            fingerprint. <code>split.py</code> decides which machine IDs are allowed into training and which one
            is held out. <code>detect.py</code> normalizes each machine, then fits the detectors and scores.{' '}
            <code>eval.py</code> turns the scores into the AUC and catch numbers. There is no web server and no
            database: training is a script that reads audio and writes a JSON file, and this page loads that
            JSON.
          </p>
          <p>
            The choice that mattered most was the per-machine normalization. Everything else I tried, richer
            features, a frozen VGGish embedding, and a CNN I trained on the machines myself, barely moved the
            held-out number or made it worse, because they were all fighting the unit-to-unit offset instead of
            removing it. Centering each machine on its own statistics removes it directly, and it is a few lines
            of code.
          </p>
          <p>
            I report catch rate at a fixed false-alarm rate rather than accuracy for a practical reason: the
            test set is mostly healthy machines, so accuracy would reward a detector that stays silent. A
            factory cares how many faults get caught and how often the alarm cries wolf, so those are the
            numbers on the page.
          </p>
          <p>
            And I wrote the bad first-cut numbers down as they came out rather than tuning until they looked
            better. Keeping the failure visible is what makes the fix mean something.
          </p>
          <p className="stack-intro">Here is what each part is built with.</p>
          <div className="stack-grid">
            {STACK.map((s) => (
              <div key={s.name} className="stack-card">
                <h3 className="stack-card__name">{s.name}</h3>
                <p className="stack-card__role">{s.role}</p>
                <p className="stack-card__how">{s.how}</p>
              </div>
            ))}
          </div>
        </StoryBeat>

        <section className="story-beat" id="usage">
          <p className="story-kicker">Running it yourself</p>
          <h2>How to reproduce every number on this page</h2>
          <p className="stack-intro">
            The repo is self-contained. These commands download the data, run the whole protocol, write the
            results into this site, and open it. Nothing on the page is hand-entered; it all comes out of this
            run.
          </p>
          <pre className="code-block">
            <code>{`git clone <repo> machineDebugger && cd machineDebugger

pip install numpy scipy scikit-learn
python scripts/run_all.py            # downloads MIMII, runs all four types
python scripts/make_web_data.py      # writes the numbers into web/src/data.ts

cd web && npm install && npm run dev  # see the walkthrough`}</code>
          </pre>
          <p>
            If a download is interrupted, just run it again, since anything already on disk is skipped. And if
            you only want to check that the split does not leak without downloading anything, run{' '}
            <code>python tests/test_eval.py</code>; it injects a leak and confirms the guard catches it.
          </p>
        </section>

        <StoryBeat
          id="next"
          kicker="Where it goes next"
          title="Using this on other problems, and what would sharpen the answer"
          caption="The discipline that transfers is the hold-out; the audio is incidental."
          visual={
            <div className="teach-card">
              <h3 className="teach-card__title">To push it further, I would add</h3>
              <ul className="next-list">
                <li>More machine IDs above all. Four is too few for a learned embedding to generalize, which is what caps {worst.type}; twenty would also let me rotate every one through the test.</li>
                <li>Rotate the held-out ID, not just {testId}, and report the spread instead of one number.</li>
                <li>The other noise levels (−6 and +6 dB), to see how much the room drives the misses.</li>
                <li>Continuous recordings, for a false-alarms-per-day figure these clipped files can&apos;t give.</li>
              </ul>
            </div>
          }
        >
          <p>
            The shape of this is not really about machines. Any time you want to flag &quot;this one looks
            wrong&quot; and you mostly have examples of normal, the same setup applies: train on the normal,
            hold out a whole real-world thing you will face (a new patient, a new sensor, a new store, a new
            customer), and measure whether the alarm still works on that held-out thing instead of on data it
            has already seen. And the lesson carries too: when a detector fails across a shift, check whether it
            is the signal that is missing or just the offset between one instance and the next.
          </p>
          <p>
            Four units per machine type is thin, and I hold out only one of them. The list beside this is what I
            would want before trusting a conclusion stronger than what is here: more machine IDs, both to
            rotate every one through the test and to give a learned embedding enough units to help {worst.type};
            the other noise levels to separate the machine from the room; and continuous recordings for a real
            false-alarm cost. With those, the same protocol would give an answer you could plan around.
          </p>
        </StoryBeat>

        <footer
          style={{
            borderTop: '1px solid var(--line-rule)',
            paddingTop: 'var(--space-6)',
            marginTop: 'var(--space-6)',
            color: 'var(--fg-low)',
            fontSize: 'var(--fs-sm)',
          }}
        >
          <p style={{ maxWidth: 'var(--measure)' }}>
            Data: MIMII 1.0 ({DATASET.cite}), Zenodo {DATASET.record}, at 0 dB. Research and portfolio software
            only. Every number here is measured by the eval and written into the page by a script; none of it is
            illustrative.
          </p>
        </footer>
      </main>
    </>
  )
}
