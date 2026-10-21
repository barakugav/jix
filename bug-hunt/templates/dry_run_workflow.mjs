// Dry-run a bug-hunt workflow script with fake agents, to check its control flow without spending
// tokens: node bug-hunt/templates/dry_run_workflow.mjs [.claude/workflows/bug-hunt.js] ['<args json>']
//
// Fake hunters return one or two findings (two hunters share a root cause, to exercise dedup);
// the fake dedup agent groups by file:line; fake verifiers cycle through every verdict, so the
// second-opinion path runs too. Prints every agent call and the workflow's return value.
// PRINT_PROMPT=<label> (e.g. hunt:reduction-driver, verify:F001, dedup) prints that agent's prompt.
import { readFileSync } from 'node:fs'

const path = process.argv[2] || '.claude/workflows/bug-hunt.js'
const args = process.argv[3] ? JSON.parse(process.argv[3]) : { runId: 'dry-run' }
const src = readFileSync(path, 'utf8').replace(/^export const meta/m, 'const meta')

const calls = []
let n = 0
const verdicts = ['confirmed', 'plausible', 'refuted', 'intended']

async function agent(prompt, opts = {}) {
  n += 1
  const label = opts.label || `agent-${n}`
  calls.push(label)
  if (process.env.PRINT_PROMPT && label === process.env.PRINT_PROMPT) console.log(`----- prompt for ${label} -----\n${prompt}\n----- end -----`)
  if (typeof prompt !== 'string' || !prompt.length) throw new Error(`${label}: empty prompt`)
  if (/undefined|\[object Object\]/.test(prompt)) throw new Error(`${label}: prompt contains "undefined" or "[object Object]"`)
  if (label.startsWith('hunt:')) {
    const i = calls.filter(c => c.startsWith('hunt:')).length
    const f = k => ({
      title: `fake bug ${k}`, severity: ['critical', 'high', 'medium', 'low'][k % 4], category: 'wrong-result',
      file: `jix/src/fake_${k % 3}.rs`, line: 10 * (k % 3), description: 'd', trigger: 't',
      expected_vs_actual: 'e', reachability: 'public-rust-api', confidence: ['high', 'medium', 'low'][k % 3],
    })
    return { findings: i % 3 === 0 ? [] : [f(i), f(i + 1)], coverage: `covered ${label}`, open_questions: [] }
  }
  if (label === 'dedup') {
    const lines = prompt.split('\n').filter(l => l.startsWith('{"i":'))
    const byLoc = new Map()
    for (const l of lines) {
      const o = JSON.parse(l)
      if (!byLoc.has(o.loc)) byLoc.set(o.loc, [])
      byLoc.get(o.loc).push(o.i)
    }
    const groups = [...byLoc.values()].map(ix => ({ indices: ix, canonical: ix[0], known: '' }))
    if (groups.length > 2) groups[0].known = 'K1'
    groups.pop() // drop one group entirely: the script must re-add its indices
    return { groups }
  }
  if (label.startsWith('verify')) {
    const v = verdicts[n % verdicts.length]
    return { verdict: v, severity: 'high', title: `verified ${label}`, reasoning: 'r', repro_path: '', writeup_path: '' }
  }
  if (label === 'report') return 'report written'
  if (label.startsWith('map:')) return { doc: 'x', units: [], suspicions: [] }
  throw new Error(`unexpected agent label ${label}`)
}
async function parallel(thunks) {
  return Promise.all(thunks.map(t => t().catch(e => { console.error('thunk error:', e.message); return null })))
}
async function pipeline(items, ...stages) {
  return Promise.all(items.map(async (item, i) => {
    let r = item
    for (const s of stages) {
      try { r = await s(r, item, i) } catch (e) { console.error('stage error:', e.message); return null }
    }
    return r
  }))
}
const log = m => console.log(`[log] ${m}`)
const phase = t => console.log(`[phase] ${t}`)

const run = new Function('agent', 'parallel', 'pipeline', 'log', 'phase', 'args', 'budget', `return (async () => {\n${src}\n})()`)
const result = await run(agent, parallel, pipeline, log, phase, args, { total: null, spent: () => 0, remaining: () => Infinity })
console.log(`[agents] ${calls.length}: ${[...new Set(calls.map(c => c.split(':')[0]))].map(k => `${k}=${calls.filter(c => c.split(':')[0] === k).length}`).join(' ')}`)
console.log(JSON.stringify(result, null, 2))
