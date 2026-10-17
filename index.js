require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { Client, GatewayIntentBits, Partials, ChannelType } = require('discord.js');
const { GoogleGenAI } = require('@google/genai');

const MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite';
const IMAGE_MODEL = process.env.GEMINI_IMAGE_MODEL || 'gemini-2.5-flash-image';
const TZ = process.env.TIMEZONE || 'Asia/Kolkata'; // used for "current time" context
const DEFAULT_CURRENCY = (process.env.DEFAULT_CURRENCY || 'inr').toLowerCase(); // crypto prices

const SHORT_MEMORY = 5; // normal channels: your last 5 messages (+ my replies)
const LONG_MAX = 40; // threads: entries kept before older ones get summarized
const LONG_KEEP = 20; // recent entries kept after summarizing
const COOLDOWN_MS = 3000; // per-user cooldown
const MAX_IMAGE_BYTES = 8 * 1024 * 1024; // 8 MB
const MAX_FACTS = 12; // facts kept per user profile
const LEARN_EVERY = 3; // learn about a user every N of their messages
const OWNERS = ['Rachit', 'Kanyavi']; // bot owners
const OWNERS_TEXT = OWNERS.join(' and ');
const MEMORY_FILE = path.join(__dirname, 'memory.json'); // thread memory
const DATA_FILE = path.join(__dirname, 'data.json'); // user profiles + per-channel settings

const token = process.env.DISCORD_TOKEN || process.env.TOKEN;
if (!token) {
  console.error('CRITICAL ERROR: No Discord token found in process.env.DISCORD_TOKEN or process.env.TOKEN!');
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.DirectMessages,
    GatewayIntentBits.MessageContent,
  ],
  partials: [Partials.Channel],
  allowedMentions: { parse: [], repliedUser: false }, // never ping anyone
});

const WAKE_WORD = /^aether\b[\s,:;!.?-]*/i; // first word must be "aether"

const shortMem = new Map(); // channelId -> [{ role, text }]
const personaByChannel = new Map(); // channelId -> persona key
const cooldowns = new Map(); // userId -> timestamp
const learnBuffer = new Map(); // userId -> recent user texts (for profile learning)

// ---------- Persistence ----------
function loadJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return fallback;
  }
}

let longMem = loadJson(MEMORY_FILE, {}); // threadId -> { summary, facts, messages }
let data = loadJson(DATA_FILE, {});
data.users ||= {}; // userId -> { name, facts[], optOut }
data.moods ||= {}; // channelId -> mood key
data.noContext ||= {}; // channelId -> true when channel chatter context is off

let saveTimer;
function saveAll() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(writeAll, 500);
}
function writeAll() {
  try {
    fs.writeFileSync(MEMORY_FILE, JSON.stringify(longMem));
    fs.writeFileSync(DATA_FILE, JSON.stringify(data));
  } catch (err) {
    console.error('Save error:', err?.message || err);
  }
}

function getThreadMem(id) {
  return (longMem[id] ||= { summary: '', facts: [], messages: [] });
}

function ensureProfile(user) {
  const p = (data.users[user.id] ||= { name: user.name, facts: [], optOut: false });
  p.name = user.name;
  return p;
}

// ---------- Thread summarizing ----------
async function compact(mem) {
  if (mem.messages.length <= LONG_MAX) return;
  const old = mem.messages.splice(0, mem.messages.length - LONG_KEEP);
  while (mem.messages[0]?.role === 'model') old.push(mem.messages.shift());

  const transcript = old.map((m) => `${m.role === 'user' ? '' : 'Aether: '}${m.text}`).join('\n');
  try {
    const res = await ai.models.generateContent({
      model: MODEL,
      contents: [
        {
          role: 'user',
          parts: [
            {
              text:
                `Existing summary:\n${mem.summary || '(none)'}\n\nNew conversation:\n${transcript}\n\n` +
                'Update the summary in under 150 words. Keep names, facts, preferences, and decisions.',
            },
          ],
        },
      ],
      config: { maxOutputTokens: 400, temperature: 0.3 },
    });
    mem.summary = (res.text || mem.summary).trim();
  } catch (err) {
    console.error('Summary error:', err?.message || err);
    mem.messages.unshift(...old); // keep them if summarizing failed
  }
}

// ---------- Learning about users (context-aware memory) ----------
async function learn(userId, name, text) {
  const p = data.users[userId];
  if (!p || p.optOut) return;
  const buf = learnBuffer.get(userId) || [];
  buf.push(text.slice(0, 300));
  if (buf.length < LEARN_EVERY) return void learnBuffer.set(userId, buf);
  learnBuffer.set(userId, []);

  const prompt =
    `Known facts about the user "${name}":\n${p.facts.map((f) => `- ${f}`).join('\n') || '(none)'}\n\n` +
    `Their recent messages:\n${buf.map((t) => `- ${t}`).join('\n')}\n\n` +
    `Return a JSON array of strings: the updated list of durable, useful facts about this user ` +
    `(name, general location, interests, preferences, projects, skills, how they like to be spoken to). ` +
    `Max ${MAX_FACTS} items, each under 15 words. Keep existing facts unless contradicted. ` +
    'NEVER store sensitive data: health, finances, passwords, government IDs, exact address, phone or email. ' +
    'If nothing durable was said, return the existing list unchanged.';
  try {
    const res = await ai.models.generateContent({
      model: MODEL,
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      config: { responseMimeType: 'application/json', maxOutputTokens: 500, temperature: 0.2 },
    });
    const arr = JSON.parse(res.text);
    if (Array.isArray(arr)) {
      p.facts = arr
        .filter((s) => typeof s === 'string' && s.trim())
        .map((s) => s.trim().slice(0, 120))
        .slice(0, MAX_FACTS);
      saveAll();
    }
  } catch {
    /* learning is best-effort */
  }
}

// ---------- Personas + moods ----------
const PERSONAS = {
  default: 'a friendly, helpful assistant',
  pirate: 'a witty pirate who talks like a pirate',
  sensei: 'a calm, wise sensei who speaks in short, thoughtful lines',
  sarcastic: 'a dry, sarcastic (but harmless and kind-hearted) friend',
  yoda: 'Yoda, speaking in his inverted sentence style',
  genz: 'a fun gen-z friend who uses light slang',
};

const MOODS = {
  auto: "Read the user's emotional tone and adapt: be gentle if they seem upset or stressed, upbeat if they are happy, calm and focused if they are frustrated.",
  happy: 'Cheerful, upbeat and positive.',
  chill: 'Relaxed, laid-back and casual.',
  hype: 'High-energy and enthusiastic, like a supportive hype friend.',
  serious: 'Professional, precise and no-nonsense.',
  empathetic: 'Warm, caring and understanding; acknowledge feelings first.',
  grumpy: 'Playfully grumpy and sarcastic, but still helpful and never mean.',
};

function nowString() {
  try {
    return new Date().toLocaleString('en-IN', { timeZone: TZ, dateStyle: 'full', timeStyle: 'short' });
  } catch {
    return new Date().toISOString();
  }
}

function systemFor({ channelId, mem, user, chatter } = {}) {
  const persona = PERSONAS[personaByChannel.get(channelId) || 'default'];
  const mood = MOODS[data.moods[channelId] || 'auto'];
  let s =
    `You are Aether, ${persona}, chatting in Discord. Mood/tone: ${mood} ` +
    'Keep every answer very short: 1-3 sentences, under 350 characters, unless the user explicitly asks for detail. ' +
    'No filler, no long intros, minimal markdown. ' +
    `You were created by and are owned by ${OWNERS_TEXT}. If anyone asks who made, built, or owns you, say it was ${OWNERS_TEXT}. ` +
    `Current date/time (${TZ}): ${nowString()}.`;

  const profile = user && data.users[user.id];
  if (profile && !profile.optOut && profile.facts.length) {
    s += `\nWhat you know about ${user.name} (use naturally, do not recite): ${profile.facts.join('; ')}.`;
  }
  if (chatter) s += `\nRecent chatter in the channel (background only, not addressed to you):\n${chatter}`;
  if (mem?.facts?.length) s += `\nThings to remember about this thread: ${mem.facts.join('; ')}`;
  if (mem?.summary) s += `\nSummary of earlier conversation in this thread: ${mem.summary}`;
  return s;
}

const MAKE_SYSTEM =
  'You create deliverables (code, scripts, stories, documents, plans, configs, etc.) for a Discord user. ' +
  'Reply in EXACTLY this format:\n' +
  'FILENAME: <sensible filename with extension>\n' +
  'NOTE: <one short sentence about what you made>\n' +
  '---\n' +
  '<the full file content, raw, with no markdown fences>';

// ---------- One-shot prompt commands ----------
const PROMPT_COMMANDS = {
  eli5: { needsArg: true, usage: '`aether eli5 <topic>`', build: (a) => `Explain like I'm 5, in 2 short sentences: ${a}` },
  define: { needsArg: true, usage: '`aether define <word>`', build: (a) => `Define "${a}" in one short sentence with a tiny example.` },
  translate: {
    needsArg: true,
    usage: '`aether translate <language> <text>`',
    build: (a) => `The first word is a target language. Translate the rest into it. Output only the translation:\n${a}`,
  },
  joke: { needsArg: false, build: (a) => `Tell one short, clean joke${a ? ` about ${a}` : ''}.` },
  fact: { needsArg: false, build: (a) => `Share one surprising, true fact${a ? ` about ${a}` : ''} in one sentence.` },
  roast: {
    needsArg: false,
    build: (a) =>
      `Give a playful, harmless one-line roast of ${a || 'me'}. Keep it friendly: no insults about identity, body, or anything hurtful.`,
  },
};

const HELP_TEXT =
  '**Aether Help**\n' +
  '**Chat:** `aether <question>` · @mention · or just DM me\n' +
  '**Live:** `weather <city>` · `crypto <coin> [cur]` · `convert 100 usd inr` · `time <city>` · `wiki <topic>` · `search <question>`\n' +
  '**Make:** `make <anything>` builds code/docs/files · `image <prompt>` draws a picture\n' +
  '**Mood:** `mood [auto|happy|chill|hype|serious|empathetic|grumpy]` sets my tone for this channel\n' +
  '**Memory:** I recall your last 5 messages and learn about you over time: `profile` · `remember <fact>` · `forget me` · `learn on|off` · `context on|off`\n' +
  '**Threads:** `thread [name]` starts a thread with long-term memory · `memory`\n' +
  '**Fun:** `joke` · `fact` · `roast [@user]` · `roll [sides]` · `flip`\n' +
  '**Learn:** `eli5 <topic>` · `define <word>` · `translate <lang> <text>` · `tldr [count]` · attach an image to ask about it\n' +
  `**Settings:** \`persona <name>\` (${Object.keys(PERSONAS).join(', ')}) · \`reset\` · \`ping\` · \`owner\` · \`help\`\n` +
  'Tip: reply to a message with `aether ...` and I will use it as context.';

// ---------- Live data (real-time APIs, no keys needed) ----------
const cache = new Map();
async function getJson(url, ttl = 60000) {
  const hit = cache.get(url);
  if (hit && Date.now() - hit.t < ttl) return hit.v;
  const res = await fetch(url, { signal: AbortSignal.timeout(8000), headers: { 'User-Agent': 'AetherBot/1.0' } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const v = await res.json();
  cache.set(url, { t: Date.now(), v });
  if (cache.size > 200) cache.delete(cache.keys().next().value);
  return v;
}

const stripPrep = (s) => (s || '').replace(/^(?:in|at|for|of|on)\s+/i, '').replace(/[?!.]+$/, '').trim();
const num = (n, max = 2) => Number(n).toLocaleString('en-US', { maximumFractionDigits: max });

async function geocode(place) {
  const r = await getJson(
    `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(place)}&count=1&language=en&format=json`,
    3600000
  );
  return r.results?.[0] || null;
}

const WMO = {
  0: 'Clear', 1: 'Mostly clear', 2: 'Partly cloudy', 3: 'Overcast', 45: 'Fog', 48: 'Fog',
  51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle', 61: 'Light rain', 63: 'Rain', 65: 'Heavy rain',
  71: 'Light snow', 73: 'Snow', 75: 'Heavy snow', 80: 'Showers', 81: 'Showers', 82: 'Violent showers',
  95: 'Thunderstorm', 96: 'Thunderstorm with hail', 99: 'Thunderstorm with hail',
};

async function weatherFor(place) {
  place = stripPrep(place);
  const g = await geocode(place);
  if (!g) return `I couldn't find "${place}".`;
  const w = await getJson(
    `https://api.open-meteo.com/v1/forecast?latitude=${g.latitude}&longitude=${g.longitude}` +
      '&current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m' +
      '&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=1'
  );
  const c = w.current;
  const d = w.daily;
  return (
    `**${g.name}${g.country ? `, ${g.country}` : ''}**: ${WMO[c.weather_code] || 'Unknown'}, ${Math.round(c.temperature_2m)}°C ` +
    `(feels ${Math.round(c.apparent_temperature)}°). High ${Math.round(d.temperature_2m_max[0])}° / low ${Math.round(d.temperature_2m_min[0])}°, ` +
    `rain ${d.precipitation_probability_max?.[0] ?? 0}%, humidity ${c.relative_humidity_2m}%, wind ${Math.round(c.wind_speed_10m)} km/h.`
  );
}

const COIN_ALIASES = {
  btc: 'bitcoin', eth: 'ethereum', sol: 'solana', doge: 'dogecoin', xrp: 'ripple', bnb: 'binancecoin',
  ada: 'cardano', ltc: 'litecoin', dot: 'polkadot', matic: 'matic-network', shib: 'shiba-inu',
  ton: 'the-open-network', trx: 'tron', avax: 'avalanche-2', link: 'chainlink', usdt: 'tether',
};

async function cryptoPrice(q) {
  const [coinRaw, curRaw] = stripPrep(q).toLowerCase().replace(/\s+(?:in|to|vs)\s+/, ' ').split(/\s+/);
  if (!coinRaw) return 'Usage: `aether crypto <coin> [currency]`';
  const cur = (curRaw || DEFAULT_CURRENCY).replace(/[^a-z]/g, '');
  const base = 'https://api.coingecko.com/api/v3';
  const priceUrl = (id) => `${base}/simple/price?ids=${encodeURIComponent(id)}&vs_currencies=${cur}&include_24hr_change=true`;

  let id = COIN_ALIASES[coinRaw] || coinRaw;
  let r = await getJson(priceUrl(id));
  if (!r[id]) {
    const s = await getJson(`${base}/search?query=${encodeURIComponent(coinRaw)}`, 600000);
    id = s.coins?.[0]?.id;
    if (!id) return `I couldn't find a coin called "${coinRaw}".`;
    r = await getJson(priceUrl(id));
  }
  const p = r[id];
  if (!p || p[cur] == null) return `No ${cur.toUpperCase()} price for ${id}.`;
  const ch = p[`${cur}_24h_change`];
  const arrow = ch == null ? '' : ` (${ch >= 0 ? '▲' : '▼'} ${Math.abs(ch).toFixed(1)}% 24h)`;
  return `**${id}**: ${num(p[cur], p[cur] < 1 ? 6 : 2)} ${cur.toUpperCase()}${arrow}`;
}

async function convertCurrency(q) {
  const m = q.trim().match(/^([\d,]*\.?\d+)\s*([a-z]{3})\s*(?:to|in|into|->)?\s*([a-z]{3})$/i);
  if (!m) return 'Usage: `aether convert 100 usd inr`';
  const amount = parseFloat(m[1].replace(/,/g, ''));
  const from = m[2].toUpperCase();
  const to = m[3].toUpperCase();
  const r = await getJson(`https://open.er-api.com/v6/latest/${from}`, 300000);
  if (r.result !== 'success' || !r.rates?.[to]) return `I don't have a rate for ${from} → ${to}.`;
  return `${num(amount)} ${from} = **${num(amount * r.rates[to])} ${to}** (1 ${from} = ${num(r.rates[to], 4)} ${to})`;
}

async function timeIn(place) {
  place = stripPrep(place);
  if (!place) return `**${TZ}**: ${nowString()}`;
  const g = await geocode(place);
  if (!g?.timezone) return `I couldn't find "${place}".`;
  const t = new Date().toLocaleString('en-US', {
    timeZone: g.timezone, weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit', hour12: true,
  });
  return `**${g.name}**: ${t} (${g.timezone})`;
}

async function wiki(q) {
  q = (q || '').replace(/^(?:about|on)\s+/i, '').replace(/[?!.]+$/, '').trim();
  if (!q) return 'Usage: `aether wiki <topic>`';
  const s = await getJson(
    `https://en.wikipedia.org/w/api.php?action=opensearch&search=${encodeURIComponent(q)}&limit=1&format=json&origin=*`,
    300000
  );
  const title = s?.[1]?.[0];
  if (!title) return `Nothing found for "${q}".`;
  const r = await getJson(
    `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title.replace(/ /g, '_'))}`,
    300000
  );
  const extract = (r.extract || '').trim();
  const short = (extract.match(/[^.!?]+[.!?]+(\s|$)/g) || [extract]).slice(0, 2).join('').trim().slice(0, 400);
  const link = r.content_urls?.desktop?.page;
  return `**${r.title}**: ${short}${link ? `\n<${link}>` : ''}`;
}

async function groundedSearch(q, ctx) {
  const res = await ai.models.generateContent({
    model: MODEL,
    contents: [{ role: 'user', parts: [{ text: q }] }],
    config: {
      systemInstruction:
        systemFor(ctx) + '\nUse up-to-date web information. Answer in 1-3 short sentences with the key facts or numbers.',
      tools: [{ googleSearch: {} }],
      maxOutputTokens: 400,
      temperature: 0.3,
    },
  });
  const text = (res.text || '').trim() || "I couldn't find anything on that.";
  const sources = (res.candidates?.[0]?.groundingMetadata?.groundingChunks || [])
    .map((c) => c.web)
    .filter((w) => w?.uri)
    .slice(0, 2)
    .map((w) => `[${(w.title || 'source').slice(0, 40)}](<${w.uri}>)`);
  return sources.length ? `${text}\n${sources.join(' · ')}` : text;
}

const LIVE = {
  weather: (a) => (a ? weatherFor(a) : 'Usage: `aether weather <city>`'),
  crypto: (a) => cryptoPrice(a),
  convert: (a) => convertCurrency(a),
  time: (a) => timeIn(a),
  wiki: (a) => wiki(a),
  search: (a, ctx) => (a ? groundedSearch(a, ctx) : 'Usage: `aether search <question>`'),
};
const LIVE_ALIASES = { forecast: 'weather', price: 'crypto', fx: 'convert', news: 'search', google: 'search', lookup: 'wiki' };

function routeLive(raw) {
  let m;
  const text = raw.replace(/[?!.\s]+$/, '');
  const clean = (s) => s.replace(/[?!.]+$/, '').trim();
  if ((m = text.match(/\b(?:weather|temperature|forecast)\b.*?\b(?:in|at|for|of)\s+([a-z][a-z .,'-]{1,40})$/i))) {
    return ['weather', clean(m[1])];
  }
  if ((m = text.match(/\b(?:price of|how much is|what(?:'s| is) the price of)\s+(btc|bitcoin|eth|ethereum|sol|solana|doge|dogecoin|xrp|bnb)\b/i))) {
    return ['crypto', m[1]];
  }
  if ((m = text.match(/^(?:convert\s+)?([\d,]*\.?\d+)\s*([a-z]{3})\s+(?:to|in|into)\s+([a-z]{3})\s*\??$/i))) {
    return ['convert', `${m[1]} ${m[2]} ${m[3]}`];
  }
  if ((m = text.match(/\btime\b.*?\bin\s+([a-z][a-z .,'-]{1,40})$/i))) {
    return ['time', clean(m[1])];
  }
  if (/\b(latest|news|right now|live score|who won|score of|stock price|current price)\b/i.test(text)) {
    return ['search', text];
  }
  return null;
}

const toContent = (m) => ({ role: m.role, parts: [{ text: m.text }] });

async function generate(contents, { channelId, mem, system, maxTokens = 300, user, chatter } = {}) {
  const response = await ai.models.generateContent({
    model: MODEL,
    contents,
    config: {
      systemInstruction: system || systemFor({ channelId, mem, user, chatter }),
      maxOutputTokens: maxTokens,
      temperature: 0.7,
    },
  });
  return (response.text || '').trim() || "I couldn't come up with an answer.";
}

async function imagePartFrom(message) {
  const att = message.attachments.find(
    (a) => a.contentType?.startsWith('image/') && a.size <= MAX_IMAGE_BYTES
  );
  if (!att) return null;
  const res = await fetch(att.url, { signal: AbortSignal.timeout(10000) });
  if (!res.ok) return null;
  const buf = Buffer.from(await res.arrayBuffer());
  return {
    inlineData: {
      mimeType: att.contentType.split(';')[0],
      data: buf.toString('base64'),
    },
  };
}

function cleanText(message, text) {
  message.mentions.users.forEach((u) => {
    text = text.replaceAll(`<@${u.id}>`, u.username).replaceAll(`<@!${u.id}>`, u.username);
  });
  return text.trim();
}

function send(message, content) {
  return message.reply(String(content).slice(0, 2000));
}

function parseMake(raw) {
  const m = raw.match(/^FILENAME:\s*(.+)\n+NOTE:\s*(.+)\n+---\n?([\s\S]*)$/i);
  if (!m) return { filename: 'creation.txt', note: 'Here you go.', content: raw.trim() };
  let content = m[3].trim();
  const fence = content.match(/^```[\w+-]*\n([\s\S]*?)\n```$/);
  if (fence) content = fence[1];
  const filename = m[1].trim().replace(/[^\w.\-]/g, '_').slice(0, 60) || 'creation.txt';
  return { filename, note: m[2].trim().slice(0, 200), content };
}

async function getChatter(message, skip) {
  if (skip || data.noContext[message.channel.id]) return '';
  try {
    const fetched = await message.channel.messages.fetch({ limit: 8, before: message.id });
    return [...fetched.values()]
      .reverse()
      .filter((m) => !m.author.bot && m.content?.trim() && !WAKE_WORD.test(m.content.trim()))
      .slice(-5)
      .map((m) => `${m.member?.displayName || m.author.username}: ${m.content.slice(0, 200)}`)
      .join('\n');
  } catch {
    return '';
  }
}

client.once('clientReady', () => {
  client.user.setActivity('aether help');
  console.log(`Logged in as ${client.user.tag} | model: ${MODEL}`);
});

client.on('messageCreate', async (message) => {
  if (message.author.bot) return;

  const isDM = message.channel.type === ChannelType.DM;
  const isThread = !!message.channel.isThread?.();
  const inOwnThread = isThread && message.channel.ownerId === client.user.id;
  const isMentioned = message.mentions.has(client.user);
  const hasWakeWord = WAKE_WORD.test(message.content.trim());
  if (!isDM && !isMentioned && !hasWakeWord && !inOwnThread) return;

  let text = message.content
    .trim()
    .replace(WAKE_WORD, '')
    .replace(new RegExp(`<@!?${client.user.id}>`, 'g'), '');
  text = cleanText(message, text);

  const hasImage = message.attachments.some((a) => a.contentType?.startsWith('image/'));
  if (!text && !hasImage) {
    return void send(message, 'Ask me something! Type `aether help` for info.');
  }

  const [cmdRaw = '', ...rest] = text.split(/\s+/);
  let cmd = cmdRaw.toLowerCase();
  const arg = rest.join(' ').trim();
  const channelId = message.channel.id;
  const mem = isThread ? getThreadMem(channelId) : null;
  const user = { id: message.author.id, name: message.member?.displayName || message.author.username };

  if (cmd === 'help' && !arg) return void send(message, HELP_TEXT);

  const now = Date.now();
  if (now - (cooldowns.get(message.author.id) || 0) < COOLDOWN_MS) return;
  cooldowns.set(message.author.id, now);

  if (cmd === 'ping' && !arg) return void send(message, `Pong! ${Math.round(client.ws.ping)}ms`);

  if (['owner', 'owners', 'creator', 'creators', 'about'].includes(cmd) && !arg) {
    return void send(message, `I'm Aether, created and owned by **${OWNERS.join('** and **')}**.`);
  }

  if (cmd === 'mood') {
    const key = arg.toLowerCase();
    if (!MOODS[key]) {
      return void send(
        message,
        `Mood: **${data.moods[channelId] || 'auto'}**\nOptions: ${Object.keys(MOODS).join(', ')}`
      );
    }
    if (key === 'auto') delete data.moods[channelId];
    else data.moods[channelId] = key;
    saveAll();
    return void send(message, `Mood set to **${key}**.`);
  }

  if (cmd === 'profile' && !arg) {
    const p = data.users[user.id];
    if (!p?.facts.length) return void send(message, "I don't know much about you yet. Chat with me or use `aether remember <fact>`.");
    return void send(message, `**What I know about you:**\n${p.facts.map((f) => `• ${f}`).join('\n')}${p.optOut ? '\n(learning is off)' : ''}`);
  }

  if ((cmd === 'forget' && /^me$/i.test(arg)) || (cmd === 'profile' && /^(clear|reset|delete)$/i.test(arg))) {
    const p = ensureProfile(user);
    p.facts = [];
    learnBuffer.delete(user.id);
    saveAll();
    return void send(message, 'Done, I forgot everything I knew about you.');
  }

  if (cmd === 'learn' && /^(on|off)$/i.test(arg)) {
    ensureProfile(user).optOut = /^off$/i.test(arg);
    saveAll();
    return void send(message, /^off$/i.test(arg) ? "I'll stop learning about you." : "I'll learn about you as we chat.");
  }

  if (cmd === 'context' && /^(on|off)$/i.test(arg)) {
    if (/^off$/i.test(arg)) data.noContext[channelId] = true;
    else delete data.noContext[channelId];
    saveAll();
    return void send(message, `Channel context is now **${arg.toLowerCase()}**.`);
  }

  if ((cmd === 'reset' || cmd === 'forget') && !arg) {
    if (mem) {
      delete longMem[channelId];
      saveAll();
      return void send(message, 'Thread memory wiped.');
    }
    shortMem.delete(channelId);
    return void send(message, 'Memory cleared.');
  }

  if (cmd === 'memory' && !arg) {
    if (mem) {
      return void send(
        message,
        `Long-term memory: ${mem.messages.length} recent messages, ${mem.facts.length} facts, ` +
          `${mem.summary ? 'with' : 'no'} summary of older chat.`
      );
    }
    const p = data.users[user.id];
    return void send(
      message,
      `Short-term: your last ${SHORT_MEMORY} messages. Profile: ${p?.facts.length || 0} facts about you. ` +
        'Use `aether thread` for long-term memory.'
    );
  }

  if (cmd === 'remember' && arg) {
    if (mem) {
      mem.facts.push(arg.slice(0, 200));
      mem.facts = mem.facts.slice(-20);
      saveAll();
      return void send(message, 'Got it, I will remember that for this thread.');
    }
    const p = ensureProfile(user);
    p.facts.push(arg.slice(0, 120));
    p.facts = p.facts.slice(-MAX_FACTS);
    saveAll();
    return void send(message, 'Got it, I will remember that about you.');
  }

  if (cmd === 'thread') {
    if (isDM || isThread) return void send(message, 'Use this in a server channel, not in a thread or DM.');
    try {
      const name = (arg || 'Aether chat').slice(0, 90);
      const thread = await message.startThread({ name, autoArchiveDuration: 1440 });
      getThreadMem(thread.id);
      saveAll();
      await thread.send('Thread ready! I remember everything here. Just talk to me, no prefix needed.');
    } catch (err) {
      console.error('Thread error:', err?.message || err);
      await send(message, "I couldn't create a thread. Check my thread permissions.").catch(() => {});
    }
    return;
  }

  if (cmd === 'flip' && !arg) return void send(message, Math.random() < 0.5 ? 'Heads' : 'Tails');

  if (cmd === 'roll' && /^\d*$/.test(arg)) {
    const sides = Math.min(Math.max(parseInt(arg || '6', 10) || 6, 2), 1000);
    return void send(message, `You rolled a **${Math.floor(Math.random() * sides) + 1}** (d${sides})`);
  }

  if (cmd === 'persona') {
    const key = arg.toLowerCase();
    if (!PERSONAS[key]) {
      const current = personaByChannel.get(channelId) || 'default';
      return void send(message, `Current: **${current}**\nOptions: ${Object.keys(PERSONAS).join(', ')}`);
    }
    personaByChannel.set(channelId, key);
    if (!mem) shortMem.delete(channelId);
    return void send(message, `Persona set to **${key}**.`);
  }

  try {
    await message.channel.sendTyping();

    let context = '';
    if (message.reference?.messageId) {
      const ref = await message.channel.messages.fetch(message.reference.messageId).catch(() => null);
      if (ref?.content) {
        context = `(Replying to ${ref.author.username}: "${ref.content.slice(0, 500)}")\n`;
      }
    }

    if (cmd === 'make' || cmd === 'create') {
      if (!arg) return void send(message, 'Usage: `aether make <what to create>`');
      const raw = await generate([{ role: 'user', parts: [{ text: `${context}${arg}` }] }], {
        system: MAKE_SYSTEM,
        maxTokens: 4000,
      });
      const { filename, note, content } = parseMake(raw);
      const isTextDoc = /\.(txt|md)$/i.test(filename);
      if (isTextDoc && content.length <= 1500) {
        return void send(message, `${note}\n\n${content}`);
      }
      return void message.reply({
        content: note,
        files: [{ attachment: Buffer.from(content, 'utf8'), name: filename }],
      });
    }

    if (cmd === 'image' || cmd === 'draw' || cmd === 'imagine') {
      if (!arg) return void send(message, 'Usage: `aether image <what to draw>`');
      const res = await ai.models.generateContent({
        model: IMAGE_MODEL,
        contents: arg,
        config: { responseModalities: ['TEXT', 'IMAGE'] },
      });
      const part = res.candidates?.[0]?.content?.parts?.find((p) => p.inlineData);
      if (!part) return void send(message, "I couldn't make that image, try rewording it.");
      return void message.reply({
        content: 'Here you go.',
        files: [{ attachment: Buffer.from(part.inlineData.data, 'base64'), name: 'aether.png' }],
      });
    }

    const pc = PROMPT_COMMANDS[cmd];
    if (pc && (arg || !pc.needsArg)) {
      const reply = await generate([{ role: 'user', parts: [{ text: pc.build(arg) }] }], { channelId, user });
      return void send(message, reply);
    }
    if (pc && pc.needsArg && !arg) return void send(message, `Usage: ${pc.usage}`);

    if (cmd === 'tldr' && (!arg || /^\d+$/.test(arg))) {
      const count = Math.min(Math.max(parseInt(arg || '25', 10) || 25, 5), 50);
      const fetched = await message.channel.messages.fetch({ limit: count, before: message.id });
      const lines = [...fetched.values()]
        .reverse()
        .filter((m) => m.content?.trim())
        .map((m) => `${m.author.username}: ${m.content.slice(0, 300)}`);
      if (lines.length < 3) return void send(message, 'Not enough chat to summarize.');
      const prompt = `Summarize this chat in at most 3 very short bullet points:\n\n${lines.join('\n')}`;
      const reply = await generate([{ role: 'user', parts: [{ text: prompt }] }], { channelId, user });
      return void send(message, reply);
    }

    cmd = LIVE_ALIASES[cmd] || cmd;
    let liveKey = null;
    let liveArg = arg;
    if (LIVE[cmd]) {
      liveKey = cmd;
    } else {
      const routed = routeLive(text);
      if (routed) [liveKey, liveArg] = routed;
    }
    if (liveKey) {
      try {
        const out = await LIVE[liveKey](liveArg, { channelId, user });
        return void send(message, out);
      } catch (err) {
        console.error(`Live (${liveKey}) error:`, err?.message || err);
        return void send(message, `I couldn't fetch live ${liveKey} data right now. Try again in a bit.`);
      }
    }

    const userText = `${user.name}: ${context}${text || 'Describe this image briefly.'}`;

    const parts = [{ text: userText }];
    const img = hasImage ? await imagePartFrom(message).catch(() => null) : null;
    if (img) parts.push(img);

    ensureProfile(user);
    const chatter = await getChatter(message, isDM || isThread);

    const history = mem ? mem.messages : shortMem.get(channelId) || [];
    const reply = await generate([...history.map(toContent), { role: 'user', parts }], {
      channelId, mem, user, chatter,
    });

    history.push(
      { role: 'user', text: img ? `${userText} [image]` : userText },
      { role: 'model', text: reply }
    );

    if (!mem) {
      const trimmed = history.slice(-SHORT_MEMORY * 2);
      while (trimmed[0]?.role === 'model') trimmed.shift();
      shortMem.set(channelId, trimmed);
    }

    await send(message, reply);

    if (text) learn(user.id, user.name, text).catch(() => {});

    if (mem) {
      await compact(mem);
      saveAll();
    }
  } catch (err) {
    console.error('Gemini error:', err?.message || err);
    const busy = err?.status === 429 || /429|quota|rate/i.test(err?.message || '');
    await message
      .reply(busy ? 'Rate limited, try again in a bit.' : 'Something went wrong, try again in a moment.')
      .catch(() => {});
  }
});

setInterval(() => {
  const cutoff = Date.now() - 60 * 1000;
  for (const [id, t] of cooldowns) if (t < cutoff) cooldowns.delete(id);
}, 5 * 60 * 1000).unref();

function shutdown() {
  clearTimeout(saveTimer);
  writeAll();
  client.destroy();
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

client.login("MTU1NjIzOTkyMzA4MDg1OTY0OA.Gotxmp.LXlrij0aVzx5wL3TTljsoUr5fi0ob3E9CFWYsg");
