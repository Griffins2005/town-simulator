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
const SPEED_PRESETS = [
  { label: '0.25×', delay: 2.4 },
  { label: '0.5×', delay: 1.2 },
  { label: '1×', delay: 0.6 },
  { label: '2×', delay: 0.3 },
  { label: '4×', delay: 0.15 },
];
const FAITH_NAME = {
  vale_covenant: 'Vale Covenant', hall_creed: 'Hall Creed',
  old_ways: 'Old Ways', unaffiliated: 'Unaffiliated'
};
const FAITH_COLOR = {
  vale_covenant: '#6b8f71', hall_creed: '#c4b59a',
  old_ways: '#8a7a4b', unaffiliated: '#7a8194'
};
let mapView = 'town';
const mapLayers = {
  residents: true, relationships: false, factions: true, faith: true,
  crossings: false, anomalies: false, terrain: false, water: true
};
let memFilter = 'all';
const frameArchive = [];
let openMenu = null;

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
function faithOf(id, frame) {
  const st = (frame && frame.agents && frame.agents[id]) || {};
  const a = AGENTS[id] || {};
  return st.faith || a.faith || 'unaffiliated';
}
function faithLabel(id, frame) {
  const a = AGENTS[id] || {};
  return a.faith_name || FAITH_NAME[faithOf(id, frame)] || 'Unaffiliated';
}
function faithLeaders(frame) {
  return ((frame && frame.faith && frame.faith.leaders) || []);
}
const LIVELIHOOD_NAME = { farmer: 'Farmer', trader: 'Trader', laborer: 'Laborer' };
function livelihoodLabel(job) {
  return LIVELIHOOD_NAME[job] || 'Laborer';
}
function isFaithLeader(id, frame) {
  return faithLeaders(frame).some(row => row.id === id);
}
function faithLeaderName(faithId, frame) {
  const row = faithLeaders(frame).find(r => r.faith === faithId);
  return row ? (row.name || nameOf(row.id)) : '';
}
const VOTE_MEMORY = {
  cast_vote: 1, proposal_resolved: 1, was_lobbied: 1, was_impeached: 1,
  was_elected: 1, vote_suspended: 1
};
function memoryKind(entry) {
  if (entry && typeof entry === 'object') return entry.kind || '';
  const m = String(entry || '').match(/\[t\d+\]\s+([a-z_]+)/i);
  return m ? m[1] : '';
}
function memoryText(entry) {
  if (entry && typeof entry === 'object') return entry.text || entry.kind || '';
  return String(entry || '');
}
function memoryTick(entry) {
  if (entry && typeof entry === 'object' && entry.tick != null) return entry.tick;
  const m = String(memoryText(entry)).match(/\[t(\d+)\]/);
  return m ? Number(m[1]) : null;
}
function memorySubject(entry) {
  if (entry && typeof entry === 'object') return entry.subject || null;
  const m = String(memoryText(entry)).match(/about\s+(agent_\d+)/);
  return m ? m[1] : null;
}
function classifyMemory(entry) {
  const kind = memoryKind(entry);
  const domain = entry && typeof entry === 'object' ? entry.domain : '';
  if (domain === 'votes' || VOTE_MEMORY[kind]) return 'votes';
  if (domain) return domain;
  if (kind) {
    if (/invent|adopt/.test(kind)) return 'inventions';
    if (/vote|proposal|lobby|expel|welcome|impeach|elect|rule/.test(kind)) return 'political';
    if (/work|trade|bankrupt|crisis/.test(kind)) return 'economic';
    return 'social';
  }
  const t = memoryText(entry).toLowerCase();
  if (/invent|pump|adopt|workshop|catalog/.test(t)) return 'inventions';
  if (/vote|proposal|lobby|faction|curfew|repeal|expel|suspend|rule_|welcome|deadlock/.test(t)) return 'political';
  if (/unrest|protest|road_closed|bridge|greenway/.test(t)) return 'political';
  if (/flood|famine|drought|pollution|food|trade|money|work|demurrage|tax|bank/.test(t)) return 'economic';
  return 'social';
}
function memoryMatchesFilter(entry, filter) {
  if (!filter || filter === 'all') return true;
  const cls = classifyMemory(entry);
  if (filter === 'political') return cls === 'political' || cls === 'votes';
  if (filter === 'votes') return cls === 'votes';
  return cls === filter;
}
function roleOf(id) {
  const t = (AGENTS[id] && AGENTS[id].traits) || {};
  const map = {
    industriousness: 'Maker', sociability: 'Organizer', generosity: 'Mediator',
    rule_respect: 'Clerk', risk_tolerance: 'Speculator'
  };
  const pairs = Object.entries(t).filter(([k]) => map[k]);
  if (!pairs.length) return 'Resident';
  pairs.sort((a, b) => b[1] - a[1]);
  return map[pairs[0][0]] || 'Resident';
}
function lawCount(rules, enacted) {
  if (Array.isArray(enacted) && enacted.length) return enacted.length;
  let n = 0;
  if (rules && ('wealth_tax_rate' in rules)) n++;
  if (rules && ('curfew_after_tick_of_day' in rules)) n++;
  if (rules && rules.festival_faith) n++;
  if (rules && rules.water_blessing) n++;
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
          'vote_suspended','lobby_succeeded','proposal_deadlocked','notoriety',
          'converted','worship_session','festival_ended','faith_leader',
          'leader_seated','leader_elected','leader_impeached','leader_stepped_down',
          'llm_fallback',
          'bankrupt','going_bankrupt','recovered'].includes(kind);
}
function eventTone(kind) {
  if (['influence_campaign','gossip','campaign_ended','call_for_expulsion','notoriety'].includes(kind)) return 'whisper';
  if (['invention','invention_adopted'].includes(kind)) return 'invent';
  if (['rule_proposed','rule_repealed','proposal_closed','vote_cast','rule_enacted',
       'lobby_succeeded','lobby_failed','member_expelled','member_arrived','member_welcomed','member_restored',
       'vote_suspended','vote_rights_restored','proposal_deadlocked','festival_ended',
       'leader_seated','leader_elected','leader_impeached','leader_stepped_down'].includes(kind)) return 'law';
  if (['worship', 'converted', 'worship_session', 'faith_leader'].includes(kind)) return 'whisper';
  if (['crisis_started','crisis_ended','corruption_scandal','market_shock',
       'bankrupt','going_bankrupt'].includes(kind)) return 'crisis';
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
  if (kind === 'worship') return 'Worship';
  if (kind === 'converted') return 'Converted';
  if (kind === 'worship_session') return 'Service';
  if (kind === 'festival_ended') return 'Festival ended';
  if (kind === 'faith_leader') return 'Faith leader';
  if (kind === 'leader_seated') return 'Leader seated';
  if (kind === 'leader_elected') return 'Leader elected';
  if (kind === 'leader_impeached') return 'Impeached';
  if (kind === 'leader_stepped_down') return 'Leader stepped down';
  if (kind === 'llm_fallback') return 'Model fallback';
  if (kind === 'bankrupt') return 'Bankrupt';
  if (kind === 'going_bankrupt') return 'Going bankrupt';
  if (kind === 'recovered') return 'Recovered';
  return String(kind || '').replace(/_/g, ' ');
}
function describeEvent(e) {
  const quoted = e.said ? ': "' + escapeHtml(e.said) + '"' : '';
  if (e.kind === 'speak') return nameOf(e.agent) + ' talks to ' + nameOf(e.to) + quoted;
  if (e.kind === 'move') {
    const dest = String(e.to || '').replace('_', ' ');
    if (e.heading) return nameOf(e.agent) + ' walks toward ' + String(e.heading).replace('_', ' ') + ' (now at ' + dest + ')';
    return nameOf(e.agent) + ' moves to ' + dest;
  }
  if (e.kind === 'road_closed') return nameOf(e.agent) + ' cannot reach ' + String(e.attempted || '').replace('_', ' ') + ' — the road is closed';
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
  if (e.kind === 'crisis_started') {
    const mag = e.intensity || e.magnitude;
    return (e.intensity ? e.intensity + ' ' : '') + e.crisis + ' begins' + (mag != null && e.intensity == null ? ' (' + mag + ')' : '');
  }
  if (e.kind === 'crisis_ended') return (e.crisis || 'crisis') + ' ends';
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
  if (e.kind === 'worship') return nameOf(e.agent) + ' worships with the ' + (FAITH_NAME[e.faith] || e.faith || 'congregation');
  if (e.kind === 'converted') return (e.name || nameOf(e.agent)) + ' joins the ' + (FAITH_NAME[e.to] || e.to || 'congregation');
  if (e.kind === 'worship_session') return (e.name || FAITH_NAME[e.faith] || e.faith) + ' in session at the ' + String(e.location || '').replace('_', ' ');
  if (e.kind === 'festival_ended') return (FAITH_NAME[e.faith] || e.faith) + ' festival ends';
  if (e.kind === 'faith_leader') return (e.agent_name || nameOf(e.agent)) + ' leads the ' + (e.name || FAITH_NAME[e.faith] || e.faith);
  if (e.kind === 'leader_seated') return (e.name || nameOf(e.agent)) + ' is seated as town leader' + (e.how ? ' (' + e.how + ')' : '');
  if (e.kind === 'leader_elected') return (e.name || nameOf(e.agent)) + ' is elected town leader';
  if (e.kind === 'leader_impeached') return (e.name || nameOf(e.agent)) + ' is impeached';
  if (e.kind === 'leader_stepped_down') return (e.name || nameOf(e.agent)) + ' steps down' + (e.reason ? ' (' + e.reason + ')' : '');
  if (e.kind === 'llm_fallback') return nameOf(e.agent) + ': model proposed ' + (e.model_proposed || 'nothing') + ', engine used ' + (e.engine_action || 'a rule action');
  if (e.kind === 'bankrupt') return (e.name || nameOf(e.agent)) + ' is bankrupt' + (e.livelihood ? ' (' + e.livelihood + ')' : '') + ' — still seated';
  if (e.kind === 'going_bankrupt') return (e.name || nameOf(e.agent)) + ' is going bankrupt';
  if (e.kind === 'recovered') return (e.name || nameOf(e.agent)) + ' recovered';
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
  set('lawsLabel', lawCount(frame.active_rules, frame.enacted_proposals));
  set('propLabel', (frame.open_proposals || []).length);
  const lead = frame.town_leader_id;
  const leadEl = document.getElementById('leaderLabel');
  if (leadEl) leadEl.textContent = lead ? nameOf(lead).split(' ')[0] : (frame.office_vacant ? 'vacant' : '—');
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
  if (typeof window.saveCheckpoint === 'function') {
    window.saveCheckpoint();
    return;
  }
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
  if (typeof window.renderCompareFromMenu === 'function') {
    window.renderCompareFromMenu();
    return;
  }
  const box = document.getElementById('compareBox');
  if (!box) return;
  box.style.display = 'block';
  box.textContent = 'Open Experiments. Save a checkpoint, then fork. That compares two towns, not two marks on one tape.';
}
function setPaused(next) {
  paused = !!next;
  document.body.classList.toggle('is-paused', paused);
  const btn = document.getElementById('pauseBtn');
  if (btn) btn.textContent = paused ? 'resume' : 'pause';
  const status = document.getElementById('status');
  if (status) status.textContent = paused ? 'paused' : 'live';
}
async function togglePause() {
  const next = paused ? 'resume' : 'pause';
  const st = await sendControl({ cmd: next });
  setPaused(st.paused);
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
async function setSpeed(delay, label) {
  speedDelay = delay;
  await sendControl({ cmd: 'speed', delay });
  const lab = document.getElementById('speedTrigger');
  if (lab) lab.textContent = label;
  document.querySelectorAll('#speedMenu .menu-row').forEach(row => {
    row.classList.toggle('is-on', Number(row.dataset.delay) === delay);
  });
}
function closeMenus() {
  document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open'));
  if (openMenu) {
    const trig = openMenu.querySelector('.menu-trigger');
    if (trig) trig.setAttribute('aria-expanded', 'false');
  }
  openMenu = null;
}
function placeMenu(menu) {
  const pop = menu.querySelector('.menu-pop');
  const trig = menu.querySelector('.menu-trigger');
  if (!pop || !trig) return;
  menu.classList.remove('up');
  const rect = trig.getBoundingClientRect();
  const spaceBelow = window.innerHeight - rect.bottom;
  const need = pop.offsetHeight || 180;
  if (menu.dataset.up === '1' || spaceBelow < need + 12) menu.classList.add('up');
}
function bindMenu(menu) {
  const trig = menu.querySelector('.menu-trigger');
  const pop = menu.querySelector('.menu-pop');
  if (!trig || !pop) return;
  trig.addEventListener('click', (ev) => {
    ev.stopPropagation();
    const was = menu.classList.contains('open');
    closeMenus();
    if (!was) {
      menu.classList.add('open');
      trig.setAttribute('aria-expanded', 'true');
      openMenu = menu;
      placeMenu(menu);
      const first = pop.querySelector('.menu-row');
      if (first) first.focus();
    } else {
      trig.focus();
    }
  });
  pop.querySelectorAll('.menu-row').forEach((row, idx, rows) => {
    row.tabIndex = 0;
    row.addEventListener('keydown', (ev) => {
      if (ev.key === 'ArrowDown') { ev.preventDefault(); rows[(idx + 1) % rows.length].focus(); }
      if (ev.key === 'ArrowUp') { ev.preventDefault(); rows[(idx - 1 + rows.length) % rows.length].focus(); }
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); row.click(); }
      if (ev.key === 'Escape') { closeMenus(); trig.focus(); }
    });
  });
}
function bindSharedControls() {
  const pause = document.getElementById('pauseBtn');
  if (pause) pause.addEventListener('click', togglePause);
  const speed = document.getElementById('speedBtn');
  if (speed && !document.getElementById('speedMenu')) speed.addEventListener('click', cycleSpeed);
  document.querySelectorAll('.menu').forEach(bindMenu);
  document.addEventListener('click', (ev) => {
    if (!ev.target.closest('.menu')) closeMenus();
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && openMenu) {
      const trig = openMenu.querySelector('.menu-trigger');
      closeMenus();
      if (trig) trig.focus();
    }
  });
  document.querySelectorAll('#speedMenu .menu-row').forEach(row => {
    row.addEventListener('click', async () => {
      await setSpeed(Number(row.dataset.delay), row.dataset.label);
      closeMenus();
    });
  });
  const exp = document.getElementById('exportTrace');
  if (exp) exp.addEventListener('click', () => {
    const blob = new Blob([JSON.stringify({ frames: frameArchive.slice(-120) }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'eidolon-trace.json';
    a.click();
    closeMenus();
  });
  const inj = document.getElementById('injectBtn');
  if (inj) inj.addEventListener('click', () => {
    const menu = document.getElementById('injectMenu');
    if (menu) menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
  });
  document.querySelectorAll('#injectMenu [data-crisis]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const kind = btn.getAttribute('data-crisis');
      const info = await sendControl({
        cmd: 'inject',
        kind,
        intensity: btn.getAttribute('data-intensity') || 'serious'
      });
      const menu = document.getElementById('injectMenu');
      if (menu) menu.style.display = 'none';
      if (typeof window.onCrisisInjected === 'function') window.onCrisisInjected(kind, info);
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
    frameArchive.push(d.frame);
    if (frameArchive.length > 240) frameArchive.shift();
    onFrame(d.frame);
  });
  es.addEventListener('reset', (ev) => {
    const d = JSON.parse(ev.data);
    if (d.static_info) {
      LOCS = d.static_info.locations || LOCS;
      AGENTS = d.static_info.agents_static || AGENTS;
      LLM_IDS = d.static_info.llm_agent_ids || [];
    }
    latestFrame = d.frame;
    frameArchive.length = 0;
    if (d.frame) frameArchive.push(d.frame);
    setPaused(true);
    if (typeof window.onTownReset === 'function') window.onTownReset(d.frame);
    else if (d.frame && onFrame) onFrame(d.frame);
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
