const palette = ['#378ADD','#1D9E75','#D85A30','#D4537E','#7F77DD','#BA7517','#888780','#639922',
                  '#A32D2D','#993556','#0F6E56','#854F0B','#534AB7','#993C1D','#185FA5','#3B6D11'];
const ringPalette = ['#56CCF2','#6FCF97','#EB5757','#BB6BD9','#F2994A','#2D9CDB'];

let LOCS = {}, AGENTS = {}, LLM_IDS = [];
let latestFrame = null, selectedId = null, paused = false;
const feedLog = [];
const commWindow = [];
const metricsHistory = [];
const forkMarks = [];
let prevTreasury = null;
let speedDelay = 0.6;

function agentColor(id) {
  const idx = parseInt(String(id).split('_')[1], 10);
  return palette[(idx || 0) % palette.length];
}
function factionColor(factionId) {
  let hash = 0;
  for (let i = 0; i < factionId.length; i++) hash = (hash * 31 + factionId.charCodeAt(i)) >>> 0;
  return ringPalette[hash % ringPalette.length];
}
function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = String(s);
  return div.innerHTML;
}
function nameOf(id) {
  if (AGENTS[id] && AGENTS[id].name) return AGENTS[id].name;
  const live = latestFrame && latestFrame.agents && latestFrame.agents[id];
  return (live && live.name) || id;
}
function factionName(factionId, frame) {
  const names = (frame && frame.faction_names) || (latestFrame && latestFrame.faction_names) || {};
  return names[factionId] || factionId;
}
function initials(id) {
  return nameOf(id).split(/\s+/).map(p => p[0]).join('').slice(0, 2);
}
function roleOf(id) {
  const t = (AGENTS[id] && AGENTS[id].traits) || {};
  const pairs = Object.entries(t);
  if (!pairs.length) return 'Resident';
  pairs.sort((a, b) => b[1] - a[1]);
  const map = {
    industriousness: 'Maker', sociability: 'Organizer', generosity: 'Mediator',
    rule_respect: 'Clerk', risk_tolerance: 'Speculator'
  };
  return map[pairs[0][0]] || 'Resident';
}
function lawCount(rules) {
  let n = 0;
  if (rules && ('wealth_tax_rate' in rules)) n++;
  if (rules && ('curfew_after_tick_of_day' in rules)) n++;
  return n;
}
function happiness(frame) {
  const reps = Object.values(frame.agents || {}).map(a => a.reputation);
  return reps.length ? reps.reduce((s, v) => s + v, 0) / reps.length : 0.5;
}
function clockFromTick(tick) {
  const day = Math.floor(tick / 24) + 1;
  const hour = tick % 24;
  return { day, hh: String(hour).padStart(2, '0'), mm: '14' };
}
function notableKind(kind) {
  return ['crisis_started','crisis_ended','corruption_scandal','faction_formed','influence_campaign',
          'campaign_ended','rule_proposed','rule_repealed','proposal_closed','market_shock',
          'invention','invention_adopted','rule_enacted','member_expelled','member_arrived','member_welcomed',
          'vote_suspended','lobby_succeeded','proposal_deadlocked','notoriety'].includes(kind);
}
function eventTone(kind) {
  if (['influence_campaign','gossip','campaign_ended','call_for_expulsion','notoriety'].includes(kind)) return 'whisper';
  if (['invention','invention_adopted'].includes(kind)) return 'invent';
  if (['rule_proposed','rule_repealed','proposal_closed','vote_cast','rule_enacted',
       'lobby_succeeded','lobby_failed','member_expelled','member_arrived','member_welcomed','member_restored',
       'vote_suspended','vote_rights_restored','proposal_deadlocked'].includes(kind)) return 'law';
  if (['crisis_started','crisis_ended','corruption_scandal','market_shock'].includes(kind)) return 'crisis';
  return '';
}
function eventTitle(kind) {
  if (kind === 'influence_campaign') return 'Whisper campaign';
  if (kind === 'invention') return 'New invention';
  if (kind === 'invention_adopted') return 'Invention adopted';
  if (kind === 'rule_proposed') return 'Law proposed';
  if (kind === 'rule_repealed') return 'Law repealed';
  if (kind === 'proposal_closed') return 'Vote closed';
  if (kind === 'lobby_succeeded') return 'Lobby';
  if (kind === 'member_expelled') return 'Expelled';
  if (kind === 'member_welcomed') return 'New member';
  if (kind === 'vote_suspended') return 'Vote suspended';
  if (kind === 'proposal_deadlocked') return 'Deadlock';
  if (kind === 'crisis_started') return 'Crisis';
  if (kind === 'market_shock') return 'Market shock';
  if (kind === 'faction_formed') return 'Faction formed';
  return String(kind || '').replace(/_/g, ' ');
}
function describeEvent(e) {
  const quoted = e.said ? ': "' + escapeHtml(e.said) + '"' : '';
  if (e.kind === 'speak') return nameOf(e.agent) + ' talks to ' + nameOf(e.to) + quoted;
  if (e.kind === 'move') return nameOf(e.agent) + ' moves to ' + String(e.to).replace('_', ' ');
  if (e.kind === 'gossip') return nameOf(e.agent) + ' gossips about ' + nameOf(e.about) + quoted;
  if (e.kind === 'lobby_succeeded') return nameOf(e.agent) + ' convinces ' + nameOf(e.to) + ' to vote ' + e.lean;
  if (e.kind === 'lobby_failed') return nameOf(e.to) + ' refuses ' + nameOf(e.agent);
  if (e.kind === 'member_expelled') return (e.name || nameOf(e.agent)) + ' is expelled';
  if (e.kind === 'member_arrived') return (e.name || nameOf(e.agent)) + ' arrives, awaiting a welcome vote';
  if (e.kind === 'member_welcomed') return (e.name || nameOf(e.agent)) + ' is welcomed onto the roll';
  if (e.kind === 'member_restored') return (e.name || nameOf(e.agent)) + ' is restored to the roll';
  if (e.kind === 'vote_suspended') return (e.name || nameOf(e.agent)) + ' loses the vote until t' + e.until;
  if (e.kind === 'vote_rights_restored') return (e.name || nameOf(e.agent)) + ' may vote again';
  if (e.kind === 'proposal_deadlocked') return 'proposal #' + e.proposal_id + ' deadlocks';
  if (e.kind === 'notoriety') return (e.name || nameOf(e.agent)) + ' is notorious';
  if (e.kind === 'call_for_expulsion') return nameOf(e.agent) + ' wants ' + nameOf(e.about) + ' expelled';
  if (e.kind === 'vote_cast') return nameOf(e.by) + ' votes ' + e.choice;
  if (e.kind === 'trade_completed') return nameOf(e.from_) + ' trades with ' + nameOf(e.to);
  if (e.kind === 'corruption_scandal') return nameOf(e.agent) + ' embezzled ' + e.skimmed + '!';
  if (e.kind === 'crisis_started') return 'crisis: ' + e.crisis + ' begins';
  if (e.kind === 'crisis_ended') return 'crisis: ' + e.crisis + ' ends';
  if (e.kind === 'rule_proposed') return nameOf(e.by) + ' proposes ' + e.rule_type;
  if (e.kind === 'rule_repealed') return 'a rule was repealed';
  if (e.kind === 'faction_joined') return nameOf(e.agent) + ' aligns with ' + escapeHtml(factionName(e.faction));
  if (e.kind === 'faction_formed') return 'faction formed: ' + escapeHtml(e.name || factionName(e.faction));
  if (e.kind === 'influence_campaign') return escapeHtml(e.name || ('whisper campaign on ' + nameOf(e.target)));
  if (e.kind === 'campaign_ended') return 'campaign ended: ' + escapeHtml(e.name || nameOf(e.target));
  if (e.kind === 'invention') return nameOf(e.agent) + ' invented ' + escapeHtml(e.name || e.invention_kind);
  if (e.kind === 'invention_adopted') return nameOf(e.agent) + ' adopted ' + escapeHtml(e.name || '');
  if (e.kind === 'market_shock') return (e.shock_kind || 'shock') + ' at the ' + e.location;
  if (e.kind === 'work') return nameOf(e.agent) + ' works';
  return String(e.kind || '').replace(/_/g, ' ');
}
function svgEl(name, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', name);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  return el;
}
async function sendControl(body) {
  const r = await fetch('/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return r.json();
}
function updateHeader(frame) {
  const m = frame.metrics || {};
  const happy = happiness(frame);
  const pop = Object.keys(frame.agents || {}).length;
  const treas = frame.treasury ?? 0;
  let delta = '';
  if (prevTreasury != null && prevTreasury !== 0) {
    const pct = ((treas - prevTreasury) / Math.abs(prevTreasury)) * 100;
    if (Math.abs(pct) >= 0.05) delta = (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%/tick';
  }
  prevTreasury = treas;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('popLabel', pop);
  set('treasuryLabel', '$' + Number(treas).toFixed(0));
  const dEl = document.getElementById('treasuryDelta');
  if (dEl) dEl.textContent = delta;
  set('giniLabel', m.gini != null ? Number(m.gini).toFixed(2) : '—');
  set('happyLabel', Math.round(happy * 100) + '%');
  set('lawsLabel', lawCount(frame.active_rules));
  set('propLabel', (frame.open_proposals || []).length);
  const lead = frame.town_leader_id;
  const leadEl = document.getElementById('leaderLabel');
  if (leadEl) leadEl.textContent = lead ? nameOf(lead).split(' ')[0] : '—';
  const crises = frame.active_crises || [];
  const levels = frame.crisis_intensity || {};
  const crisisBits = crises.map((c) => {
    const mag = levels[c];
    const label = mag >= 0.8 ? 'severe' : mag >= 0.5 ? 'serious' : mag != null ? 'mild' : '';
    return label ? c + ' ' + label : c;
  });
  set('crisisLabel', crisisBits.length ? crisisBits.join(', ') : '0');
  const anomalies = (m.anomaly_flags || []).length;
  set('alertLabel', anomalies);
  const crisisWrap = document.getElementById('crisisVital');
  if (crisisWrap) crisisWrap.classList.toggle('warn', (frame.active_crises || []).length > 0);
  const alertWrap = document.getElementById('alertVital');
  if (alertWrap) alertWrap.classList.toggle('warn', anomalies > 0);
  const clk = clockFromTick(frame.tick);
  set('clockTime', clk.hh + ':' + clk.mm);
  set('clockDay', 'Day ' + clk.day);
  set('tickNum', 'Tick ' + frame.tick);
  const status = document.getElementById('status');
  if (status) status.textContent = paused ? 'paused' : 'live';
}
function recordMetrics(frame) {
  const m = frame.metrics || {};
  metricsHistory.push({
    tick: frame.tick,
    gini: m.gini || 0,
    entropy: m.location_entropy || 0,
    anomaly: m.anomaly_score || 0,
    location_counts: m.location_counts || {},
    domains: m.domains || { social: 0, political: 0, economic: 0 }
  });
  if (metricsHistory.length > 200) metricsHistory.shift();
  (frame.events || []).forEach(e => {
    if (['speak','gossip','trade_completed'].includes(e.kind)) {
      commWindow.push({ tick: frame.tick, kind: e.kind, a: e.agent || e.from_, b: e.to || e.about });
    }
  });
  while (commWindow.length && frame.tick - commWindow[0].tick > 20) commWindow.shift();
}
function ingestFeed(frame) {
  (frame.events || []).forEach(e => {
    const tone = eventTone(e.kind);
    const html = '<div class="evt ' + tone + '"><div class="k">' + escapeHtml(eventTitle(e.kind)) +
      '</div>' + describeEvent(e) + '</div>';
    feedLog.unshift({ html, notable: notableKind(e.kind), kind: e.kind, tick: frame.tick, crossings: (frame.metrics || {}).crossings || [] });
  });
  const crosses = (frame.metrics || {}).crossings || [];
  if (crosses.length) {
    feedLog.unshift({
      html: '<div class="evt cross"><div class="k">Domain crossing</div>' +
        escapeHtml(crosses.join(', ')) + ' — social, political, and economic activity coupled this tick.</div>',
      notable: true, kind: 'domain_crossing', tick: frame.tick
    });
  }
  feedLog.splice(80);
}
function markFork() {
  if (!latestFrame) return;
  forkMarks.push({
    tick: latestFrame.tick,
    gini: (latestFrame.metrics || {}).gini,
    happy: happiness(latestFrame),
    crises: (latestFrame.active_crises || []).join(',') || 'none',
    inventions: (latestFrame.inventions || []).map(i => i.name).join(', ') || 'none'
  });
  if (forkMarks.length > 2) forkMarks.shift();
  const note = document.getElementById('forkNote');
  if (note) note.textContent = 'Marked: ' + forkMarks.map(m => 't' + m.tick).join(' vs ');
}
function compareForks() {
  const box = document.getElementById('compareBox');
  if (!box) return;
  if (forkMarks.length < 2) {
    box.style.display = 'block';
    box.textContent = 'Mark two ticks first. This compares recorded vitals; it does not rewind the live engine.';
    return;
  }
  const [a, b] = forkMarks;
  box.style.display = 'block';
  box.innerHTML = '<b>Compare marked ticks</b><br>t' + a.tick + ' → Gini ' + a.gini + ', happiness ' +
    (a.happy * 100).toFixed(0) + '%, crises ' + a.crises + ', inventions ' + a.inventions +
    '<br>t' + b.tick + ' → Gini ' + b.gini + ', happiness ' + (b.happy * 100).toFixed(0) +
    '%, crises ' + b.crises + ', inventions ' + b.inventions;
}
async function togglePause() {
  const next = paused ? 'resume' : 'pause';
  const st = await sendControl({ cmd: next });
  paused = st.paused;
  const btn = document.getElementById('pauseBtn');
  if (btn) btn.textContent = paused ? 'resume' : 'pause';
}
async function cycleSpeed() {
  const steps = [0.3, 0.6, 1.2];
  const labels = ['2x', '1x', '0.5x'];
  let i = steps.indexOf(speedDelay);
  i = (i + 1) % steps.length;
  speedDelay = steps[i];
  await sendControl({ cmd: 'speed', delay: speedDelay });
  const lab = document.getElementById('speedBtn');
  if (lab) lab.textContent = 'speed ' + labels[i];
}
function bindSharedControls() {
  const pause = document.getElementById('pauseBtn');
  if (pause) pause.addEventListener('click', togglePause);
  const speed = document.getElementById('speedBtn');
  if (speed) speed.addEventListener('click', cycleSpeed);
  const inj = document.getElementById('injectBtn');
  if (inj) inj.addEventListener('click', () => {
    const menu = document.getElementById('injectMenu');
    if (menu) menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
  });
  document.querySelectorAll('#injectMenu [data-crisis]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      await sendControl({
        cmd: 'inject',
        kind: btn.getAttribute('data-crisis'),
        intensity: btn.getAttribute('data-intensity') || 'serious'
      });
      const menu = document.getElementById('injectMenu');
      if (menu) menu.style.display = 'none';
    });
  });
  const mark = document.getElementById('markFork');
  if (mark) mark.addEventListener('click', markFork);
  const cmp = document.getElementById('compareFork');
  if (cmp) cmp.addEventListener('click', compareForks);
}
function connectStream(onFrame) {
  const es = new EventSource('/stream');
  const status = document.getElementById('status');
  es.addEventListener('tick_started', (ev) => {
    const d = JSON.parse(ev.data);
    if (status && !paused) status.textContent = 'tick ' + d.tick + ' thinking';
  });
  es.addEventListener('frame', (ev) => {
    const d = JSON.parse(ev.data);
    latestFrame = d.frame;
    Object.entries((d.frame && d.frame.agents) || {}).forEach(([id, st]) => {
      if (st && st.name && !AGENTS[id]) {
        AGENTS[id] = { name: st.name, traits: {}, decider_kind: 'RuleBasedDecider' };
      }
    });
    if (status) status.textContent = paused ? 'paused' : 'live';
    onFrame(d.frame);
  });
  es.onerror = () => { if (status) status.textContent = 'retrying'; };
}
async function bootLab(onReady) {
  const info = await (await fetch('/static_info')).json();
  LOCS = info.locations; AGENTS = info.agents_static; LLM_IDS = info.llm_agent_ids;
  bindSharedControls();
  if (onReady) onReady();
  connectStream(window.renderFrame);
}
