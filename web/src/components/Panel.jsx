import React from 'react'

export function Panel({ title, sub, right, children, bodyClass = '', style }) {
  return (
    <section className="panel" style={style}>
      <header className="panel-head">
        <span className="panel-title">{title}</span>
        {right}
        {sub ? <span className="panel-sub">{sub}</span> : null}
      </header>
      <div className={`panel-body ${bodyClass}`}>{children}</div>
    </section>
  )
}

export function Stat({ k, v, s, tone }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className={`v ${tone || ''}`}>{v}</div>
      {s ? <div className="s">{s}</div> : null}
    </div>
  )
}
