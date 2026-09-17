#!/usr/bin/env node
'use strict';

// 语音问答端到端回放：在 Node 里执行 PWA 的真实内联脚本，对着正在运行的网关，
// 按浏览器 ScriptProcessor 的节奏把 16 kHz s16le PCM 喂进 onaudioprocess，
// 音频放完后继续喂静音，直到页面自己的 VAD 停止录音。
//
// 量的是"这份前端脚本 + 真实网关/模型"，不是真实浏览器：没有麦克风、音频栈、
// 页面调度和局域网，不能当作真机 P95（architecture.md §6.1）。
// 输出不含转写或答案文本——回放素材可能是私有录音。
//
// 用法：node bench/bench_pwa_voice.cjs --base http://127.0.0.1:8080 --pcm clip.s16le \
//         --language en [--runs 3] [--revision <git rev>] [--json out.json]

const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, token, i, all) => {
  if (token.startsWith('--')) pairs.push([token.slice(2), all[i + 1]]);
  return pairs;
}, []));
for (const required of ['base', 'pcm', 'language']) {
  if (!args[required]) throw new Error(`missing --${required}`);
}
const runs = Number(args.runs || 1);
const repo = path.resolve(__dirname, '..');
const html = args.revision
  ? execFileSync('git', ['show', `${args.revision}:apps/pwa/index.html`], {cwd: repo, encoding: 'utf8'})
  : fs.readFileSync(path.join(repo, 'apps/pwa/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

// Chrome/Safari 的 ScriptProcessor(8192) 在 48 kHz 下约每 170.6 ms 回调一次；
// 这里直接以 16 kHz 喂等时长的 2730 个样本，使 pcm16() 的重采样比为 1。
const SAMPLE_RATE = 16000;
const BLOCK = 2730;
const TIMEOUT_MS = 180000;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function runOnce(samples) {
  const elements = new Map();
  for (const id of ['f', 'language', 'q', 'go', 'mic', 'cancel', 'transcript', 'status', 'answer', 'sources', 'meta']) {
    elements.set(`#${id}`, {
      disabled: false, textContent: '', innerHTML: '', className: '', value: '',
      handlers: {}, addEventListener(name, fn) { this.handlers[name] = fn; }, appendChild() {},
    });
  }
  let node = null;
  const counts = {partial: 0, final: 0, partialBytes: []};
  const ctx = {
    sampleRate: SAMPLE_RATE, destination: {}, close() {},
    createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}}),
  };
  const sandbox = {
    document: {querySelector: selector => elements.get(selector), createElement: () => ({appendChild() {}})},
    navigator: {mediaDevices: {getUserMedia: async () => ({getTracks: () => [{stop() {}}]})}},
    AudioContext: class { constructor() { return ctx; } },
    fetch: (url, options = {}) => {
      if (url.endsWith('/chunks')) {
        const body = JSON.parse(options.body);
        counts[body.final ? 'final' : 'partial'] += 1;
        if (!body.final) counts.partialBytes.push(Buffer.from(body.pcm_s16le_b64, 'base64').length);
      }
      return fetch(args.base + url, options);
    },
    performance, TextDecoder, TextEncoder, btoa, Int16Array, Uint8Array, Float32Array, Math, JSON, Error, Promise,
    String, Array, Object, console,
  };
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox, {filename: 'apps/pwa/index.html'});
  elements.get('#language').value = args.language;

  await elements.get('#mic').handlers.click();
  if (!node || !node.onaudioprocess) throw new Error(`recording did not start: ${elements.get('#status').textContent}`);
  const started = performance.now();
  let offset = 0, stoppedAtAudioS = null;
  while (node.onaudioprocess) {
    const due = started + offset / SAMPLE_RATE * 1000;
    await sleep(Math.max(0, due - performance.now()));
    const block = offset < samples.length
      ? samples.subarray(offset, offset + BLOCK)
      : new Float32Array(BLOCK);  // trailing silence lets the page's own VAD end the recording
    node.onaudioprocess({inputBuffer: {getChannelData: () => block}});
    offset += BLOCK;
    if (!node.onaudioprocess) stoppedAtAudioS = offset / SAMPLE_RATE;
    if (offset > samples.length + SAMPLE_RATE * 5 && node.onaudioprocess) {
      await elements.get('#mic').handlers.click();  // VAD never fired: stop manually
      stoppedAtAudioS = offset / SAMPLE_RATE;
    }
  }

  const deadline = performance.now() + TIMEOUT_MS;
  const meta = elements.get('#meta');
  while (!meta.textContent.includes('服务端') && performance.now() < deadline) {
    if (elements.get('#status').className === 'err') break;
    await sleep(50);
  }
  const text = meta.textContent;
  const client = text.match(/本机：(VAD 结束|停止录音)→首个正文 (\d+) ms(?:（其中排空 partial 至发出 final (\d+) ms）)?/);
  const server = text.match(/服务端（自 (\S+?)，ms）：(.*)/);
  const tokens = text.match(/上下文 (\S+) tok/);
  return {
    error: elements.get('#status').className === 'err' ? elements.get('#status').textContent : null,
    audio_s: +(samples.length / SAMPLE_RATE).toFixed(2),
    stopped_at_audio_s: stoppedAtAudioS && +stoppedAtAudioS.toFixed(2),
    stop_reason: client ? (client[1] === 'VAD 结束' ? 'vad' : 'manual') : null,
    partial_uploads: counts.partial,
    final_uploads: counts.final,
    max_partial_bytes: counts.partialBytes.length ? Math.max(...counts.partialBytes) : 0,
    client_ms: client ? {stop_to_first_text: +client[2], stop_to_final_sent: client[3] === undefined ? null : +client[3]} : null,
    prompt_tokens: tokens ? tokens[1] : null,
    server_stages: server ? {
      origin: server[1],
      ms: Object.fromEntries(server[2].split(' · ').map(pair => {
        const [name, value] = pair.split(' ');
        return [name, Number(value)];
      })),
    } : null,
  };
}

(async () => {
  const bytes = fs.readFileSync(args.pcm);
  const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.length / 2));
  const samples = Float32Array.from(pcm, v => v / 32768);
  const results = [];
  for (let i = 0; i < runs; i++) {
    const result = await runOnce(samples);
    results.push(result);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  }
  if (args.json) {
    fs.writeFileSync(args.json, JSON.stringify({
      revision: args.revision || 'working-tree', base: args.base, language: args.language,
      pcm: path.basename(args.pcm), runs: results,
    }, null, 2));
  }
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
