import { Children, cloneElement, isValidElement, useState, type ReactNode } from 'react'

type Props = {
  id: string
  title: string
  kicker?: string
  children: ReactNode
  visual?: ReactNode
  caption: string
}

export function StoryBeat({ id, title, kicker, children, visual, caption }: Props) {
  const [replayKey, setReplayKey] = useState(0)
  const shown = Children.map(visual, (child, i) =>
    isValidElement(child) ? cloneElement(child, { key: `${replayKey}-${i}` }) : child,
  )

  return (
    <section className="story-beat" id={id}>
      <div className="story-beat__grid">
        <div className="story-beat__copy">
          {kicker ? <p className="story-kicker">{kicker}</p> : null}
          <h2>{title}</h2>
          <div className="story-prose">{children}</div>
        </div>
        <div className="story-beat__panel">
          <div className="story-window">
            {shown ? (
              <div className="story-viz" key={replayKey}>
                {shown}
              </div>
            ) : null}
            <div className="story-window__bar">
              <p className="meta story-caption">{caption}</p>
              <button type="button" onClick={() => setReplayKey((k) => k + 1)}>
                Replay
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
