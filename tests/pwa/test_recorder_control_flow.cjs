#!/usr/bin/env node
'use strict';

// Execute the PWA's inline script with a deliberately small browser facade.
// These are control-flow tests, not a claim of real-browser/audio compatibility.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const repo = path.resolve(__dirname, '../..');
const revision = process.argv[2];
const html = revision
  ? execFileSync('git', ['show', `${revision}:apps/pwa/index.html`], {cwd: repo, encoding: 'utf8'})
  : fs.readFileSync(path.join(repo, 'apps/pwa/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

const tick = () => new Promise(resolve => setImmediate(resolve));
const response = body => ({ok: true, json: async () => body});
const streamResponse = (sse = '') => {
  let sent = false;
  return {ok: true, body: {getReader: () => ({read: async () => {
    if (sent || !sse) return {done: true};
    sent = true; return {done: false, value: new TextEncoder().encode(sse)};
  }})}};
};

function deferred() {
  let resolve, reject;
  const promise = new Promise((ok, bad) => { resolve = ok; reject = bad; });
  return {promise, resolve, reject};
}

const timedStream = (clock, parts) => {
  let i = 0;
  return {ok: true, body: {getReader: () => ({read: async () => {
    if (i >= parts.length) return {done: true};
    const [at, sse] = parts[i++];
    clock.t = at;
    return {done: false, value: new TextEncoder().encode(sse)};
  }})}};
};

function boot({fetch, getUserMedia, context, clock = {t: 0}}) {
  const elements = new Map();
  for (const id of ['f', 'language', 'q', 'go', 'mic', 'cancel', 'transcript', 'status', 'answer', 'sources', 'meta']) {
    elements.set(`#${id}`, {
      disabled: false, textContent: '', innerHTML: '', className: '', value: '',
      handlers: {}, addEventListener(name, fn) { this.handlers[name] = fn; },
      appendChild() {},
    });
  }
  const sandbox = {
    document: {querySelector: selector => elements.get(selector), createElement: () => ({})},
    navigator: {mediaDevices: {getUserMedia}},
    AudioContext: context.constructor,
    fetch,
    performance: {now: () => clock.t},
    TextDecoder,
    btoa: value => Buffer.from(value, 'binary').toString('base64'),
    Int16Array, Uint8Array, Math, JSON, Error, Promise, console,
  };
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox, {filename: 'apps/pwa/index.html'});
  elements.get('#language').value = 'zh';
  return {
    click: () => elements.get('#mic').handlers.click(),
    cancel: () => elements.get('#cancel').handlers.click(),
    submit: () => elements.get('#f').handlers.submit({preventDefault() {}}),
    element: id => elements.get(`#${id}`),
    chooseLanguage: value => {
      const select = elements.get('#language');
      if (!select.disabled) select.value = value;
    },
  };
}

async function initializationFailureReleasesEverything() {
  const track = {stopped: false, stop() { this.stopped = true; }};
  const stream = {getTracks: () => [track]};
  const ctx = {closed: false, sampleRate: 48000, destination: {}, close() { this.closed = true; },
    createMediaStreamSource: () => source, createScriptProcessor: () => node};
  const node = {disconnect() {}, connect() {}, onaudioprocess: null};
  const source = {connect() { throw new Error('connect failed after session creation'); }};
  class FakeAudioContext {
    constructor() { return ctx; }
  }
  ctx.constructor = FakeAudioContext;
  const calls = [];
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async (url, options = {}) => {
      calls.push([url, options.method || 'GET']);
      if (url === '/v1/transcriptions') return response({transcript_id: 'created-id'});
      if (url === '/v1/transcriptions/created-id') return response({});
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  await app.click();
  await tick();
  assert.equal(track.stopped, true, 'failed initialization must stop microphone tracks');
  assert.equal(ctx.closed, true, 'failed initialization must close AudioContext');
  assert.deepEqual(calls, [
    ['/v1/transcriptions', 'POST'],
    ['/v1/transcriptions/created-id', 'DELETE'],
  ], 'a created server session must be cancelled when later initialization fails');
}

async function audioContextFailureReleasesGrantedMicrophone() {
  const track = {stopped: false, stop() { this.stopped = true; }};
  const stream = {getTracks: () => [track]};
  const failingContext = {constructor: class { constructor() { throw new Error('AudioContext failed'); } }};
  const app = boot({
    getUserMedia: async () => stream,
    context: failingContext,
    fetch: async () => { throw new Error('fetch must not be reached'); },
  });
  await app.click();
  assert.equal(track.stopped, true, 'AudioContext failure after permission must release microphone tracks');
}

async function stoppingClosesCallbackBeforeAwaitingQueue() {
  const track = {stopped: false, stop() { this.stopped = true; }};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {closed: false, sampleRate: 16000, destination: {}, close() { this.closed = true; }, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const partial = deferred();
  const chunks = [];
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async (url, options = {}) => {
      if (url === '/v1/transcriptions') return response({transcript_id: 'one'});
      if (url === '/v1/transcriptions/one/chunks') {
        const payload = JSON.parse(options.body); chunks.push(payload);
        if (!payload.final) await partial.promise;
        return response({stream_url: '/stream'});
      }
      if (url === '/stream' || url === '/v1/transcriptions/one') return streamResponse();
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  await app.click();
  const audio = value => ({inputBuffer: {getChannelData: () => new Float32Array([value])}});
  node.onaudioprocess(audio(0.1)); // held A
  node.onaudioprocess(audio(0.2)); // queue partial A, held B
  await tick();
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].final, false);
  const stopping = app.click();
  assert.equal(node.onaudioprocess, null, 'stop must close callback synchronously before awaiting');
  assert.equal(track.stopped, true, 'stop must release microphone tracks before a pending upload resolves');
  assert.equal(ctx.closed, true, 'stop must close AudioContext before a pending upload resolves');
  assert.equal(chunks.length, 1, 'stop must drain, not cancel, an already-started partial');
  const frozen = chunks.length;
  if (node.onaudioprocess) node.onaudioprocess(audio(0.3));
  assert.equal(chunks.length, frozen, 'post-stop callback must not append a partial');
  partial.resolve();
  await stopping; await tick(); await tick();
  assert.equal(chunks.length, 2, 'only the held final follows the earlier partial');
  assert.equal(chunks[1].final, true);
  assert.equal(chunks.filter(chunk => chunk.final).length, 1, 'final must be unique');
}

async function stoppingBeforeFirstAudioCallbackCancelsServerSession() {
  const track = {stopped: false, stop() { this.stopped = true; }};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {closed: false, sampleRate: 16000, destination: {}, close() { this.closed = true; }, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const calls = [];
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async (url, options = {}) => {
      calls.push([url, options.method || 'GET']);
      if (url === '/v1/transcriptions') return response({transcript_id: 'empty'});
      if (url === '/v1/transcriptions/empty') return response({});
      if (url === '/v1/transcriptions/empty/chunks') throw new Error('an empty recording must not upload a chunk');
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  await app.click();
  await app.click();
  for (let i = 0; i < 3; i++) await tick();
  assert.equal(track.stopped, true, 'zero-PCM stop must still release microphone hardware');
  assert.equal(ctx.closed, true, 'zero-PCM stop must still close AudioContext');
  assert.equal(node.onaudioprocess, null, 'zero-PCM stop must close the audio callback');
  assert.deepEqual(calls, [
    ['/v1/transcriptions', 'POST'],
    ['/v1/transcriptions/empty', 'DELETE'],
  ], 'a session without a final chunk must be explicitly deleted');
}

async function chosenLanguageFlowsToTextAndTranscriptionAndLocksDuringRecording() {
  const track = {stop() {}};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {sampleRate: 16000, destination: {}, close() {}, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const calls = [], creation = deferred();
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async (url, options = {}) => {
      calls.push([url, options.method || 'GET', options.body]);
      if (url === '/v1/answers') return response({stream_url: '/answer-stream'});
      if (url === '/v1/transcriptions') return creation.promise;
      if (url === '/answer-stream' || url === '/v1/transcriptions/language-id') return streamResponse();
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  app.element('language').value = 'en';
  app.element('q').value = 'How do I stop Spring Boot?';
  await app.submit();
  assert.deepEqual(JSON.parse(calls[0][2]), {question: 'How do I stop Spring Boot?', language: 'en'});
  const starting = app.click(); await tick();
  assert.deepEqual(JSON.parse(calls[2][2]), {language: 'en'});
  assert.equal(app.element('language').disabled, true, 'language locks before the creation response returns');
  app.chooseLanguage('zh');
  assert.equal(app.element('language').value, 'en', 'a user cannot change the visible language while creation is pending');
  creation.resolve(response({transcript_id: 'language-id'}));
  await starting;
  assert.equal(app.element('language').value, 'en', 'created ASR language and visible selection stay aligned');
  await app.click(); await tick(); await tick();
  assert.equal(app.element('language').disabled, false, 'language unlocks after the recording finishes');
}

async function failedCreationUnlocksLanguage() {
  const track = {stop() {}};
  const stream = {getTracks: () => [track]};
  const ctx = {close() {}, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => ({onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async () => ({ok: false, status: 503}),
  });
  await app.click();
  assert.equal(app.element('language').disabled, false, 'failed creation must restore language selection');
}

async function finalTranscriptStartsOneLanguageBoundAnswer() {
  const track = {stop() {}};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {sampleRate: 16000, destination: {}, close() {}, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const answers = [], chunks = [];
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async (url, options = {}) => {
      if (url === '/v1/transcriptions') return response({transcript_id: 'handoff'});
      if (url === '/v1/transcriptions/handoff/chunks') {
        chunks.push(JSON.parse(options.body));
        return response({stream_url: '/final-transcript'});
      }
      if (url === '/final-transcript') return streamResponse(
        'data: {"type":"transcript","text":"How does Kafka","final":false,"sequence":1}\n\n' +
        'data: {"type":"transcript","text":"How does Kafka compaction work?","final":true,"sequence":2}\n\n'
      );
      if (url === '/v1/answers') {
        answers.push(JSON.parse(options.body));
        return response({stream_url: '/answer-stream'});
      }
      if (url === '/answer-stream') return streamResponse();
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  app.chooseLanguage('en');
  await app.click();
  node.onaudioprocess({inputBuffer: {getChannelData: () => new Float32Array([0.2])}});
  await app.click();
  for (let i = 0; i < 4; i++) await tick();
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].final, true);
  assert.equal(answers.length, 1, 'partial transcript must not start an answer');
  assert.deepEqual(answers, [{question: 'How does Kafka compaction work?', language: 'en', transcript_id: 'handoff'}],
    'a voice answer names its transcript so the server can join ASR stage marks');
  assert.equal(app.element('q').value, 'How does Kafka compaction work?');
}

async function cancellationReleasesHardwareAndNeverFinalizes() {
  const track = {stopped: false, stop() { this.stopped = true; }};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {closed: false, sampleRate: 16000, destination: {}, close() { this.closed = true; }, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const partial = deferred(), chunks = [], deleted = [];
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async (url, options = {}) => {
      if (url === '/v1/transcriptions') return response({transcript_id: 'cancel-me'});
      if (url === '/v1/transcriptions/cancel-me/chunks') {
        chunks.push(JSON.parse(options.body)); await partial.promise; return response({stream_url: '/partial'});
      }
      if (url === '/v1/transcriptions/cancel-me') { deleted.push(options.method); return response({}); }
      if (url === '/partial') return streamResponse();
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  await app.click();
  const audio = value => ({inputBuffer: {getChannelData: () => new Float32Array([value])}});
  node.onaudioprocess(audio(0.1)); node.onaudioprocess(audio(0.2));
  await tick();
  app.cancel();
  assert.equal(track.stopped, true); assert.equal(ctx.closed, true);
  assert.deepEqual(deleted, ['DELETE']);
  assert.equal(app.element('cancel').disabled, true);
  partial.resolve(); for (let i = 0; i < 3; i++) await tick();
  assert.equal(chunks.length, 1, 'cancel must not append a final chunk');
  assert.equal(chunks[0].final, false);
}

async function cancellationDiscardsLateTranscriptEvents() {
  const track = {stop() {}};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {sampleRate: 16000, destination: {}, close() {}, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const partial = deferred(), answers = [];
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async (url, options = {}) => {
      if (url === '/v1/transcriptions') return response({transcript_id: 'late-event'});
      if (url === '/v1/transcriptions/late-event/chunks') {
        await partial.promise;
        return response({stream_url: '/late-transcript'});
      }
      if (url === '/v1/transcriptions/late-event') return response({});
      if (url === '/late-transcript') return streamResponse(
        'data: {"type":"transcript","text":"discarded final","final":true,"sequence":1}\n\n'
      );
      if (url === '/v1/answers') {
        answers.push(JSON.parse(options.body));
        return response({stream_url: '/answer-stream'});
      }
      if (url === '/answer-stream') return streamResponse();
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  await app.click();
  const audio = value => ({inputBuffer: {getChannelData: () => new Float32Array([value])}});
  node.onaudioprocess(audio(0.1)); node.onaudioprocess(audio(0.2));
  await tick();
  const transcriptBeforeCancel = app.element('transcript').textContent;
  app.cancel();
  partial.resolve();
  for (let i = 0; i < 5; i++) await tick();
  assert.equal(app.element('transcript').textContent, transcriptBeforeCancel,
    'a cancelled recording must discard a late transcript SSE instead of displaying it');
  assert.equal(app.element('q').value, '', 'a late final transcript must not overwrite the question input');
  assert.deepEqual(answers, [], 'a late final transcript must not start an answer');
}

async function vadStopsOnlyAfterSpeechAndAccumulatedSilence() {
  const track = {stopped: false, stop() { this.stopped = true; }};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {closed: false, sampleRate: 16000, destination: {}, close() { this.closed = true; }, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const chunks = [];
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    fetch: async (url, options = {}) => {
      if (url === '/v1/transcriptions') return response({transcript_id: 'vad'});
      if (url === '/v1/transcriptions/vad/chunks') {
        chunks.push(JSON.parse(options.body));
        return response({stream_url: '/vad-stream'});
      }
      if (url === '/vad-stream') return streamResponse();
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  await app.click();
  const audio = value => ({inputBuffer: {getChannelData: () => new Float32Array(9600).fill(value)}}); // 600 ms at 16 kHz
  node.onaudioprocess(audio(0));
  node.onaudioprocess(audio(0));
  assert.equal(track.stopped, false, 'pre-speech silence must not end a newly opened recording');
  node.onaudioprocess(audio(0.1));
  node.onaudioprocess(audio(0));
  assert.equal(track.stopped, false, '600 ms of post-speech silence is below the VAD stop window');
  node.onaudioprocess(audio(0));
  assert.equal(track.stopped, true, 'speech followed by 1.2 s of silence must stop the microphone');
  assert.equal(ctx.closed, true, 'VAD stop must reuse the immediate hardware-release path');
  assert.equal(node.onaudioprocess, null, 'VAD stop must close the callback before draining uploads');
  for (let i = 0; i < 8; i++) await tick();
  assert.equal(chunks.filter(chunk => chunk.final).length, 1, 'VAD stop must emit one final upload');
  assert.equal(chunks.at(-1).final, true, 'the VAD final upload must remain ordered after partials');
}

async function voiceAnswerReportsClientStopToFirstTextAndServerStages() {
  const track = {stop() {}};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {sampleRate: 16000, destination: {}, close() {}, createMediaStreamSource: () => ({connect() {}}),
    createScriptProcessor: () => (node = {onaudioprocess: null, connect() {}, disconnect() {}})};
  class FakeAudioContext { constructor() { return ctx; } }
  ctx.constructor = FakeAudioContext;
  const clock = {t: 0}, partial = deferred(), answers = [];
  const doneEvent = stages => `data: ${JSON.stringify({type: 'done', ttft_s: 0.5, total_s: 1, decode_tps: 10,
    prompt_tokens: 5, prefilled_tokens: 5, template_version: 'v', stages})}\n\n`;
  const app = boot({
    getUserMedia: async () => stream,
    context: ctx,
    clock,
    fetch: async (url, options = {}) => {
      if (url === '/v1/transcriptions') return response({transcript_id: 'timed'});
      if (url === '/v1/transcriptions/timed/chunks') {
        const {final} = JSON.parse(options.body);
        if (!final) { await partial.promise; return response({stream_url: '/partial-stream'}); }
        return response({stream_url: '/final-stream'});
      }
      if (url === '/partial-stream') return streamResponse();
      if (url === '/final-stream') return streamResponse(
        'data: {"type":"transcript","text":"Kafka 怎么排查重复消费","final":true,"sequence":3}\n\n');
      if (url === '/v1/answers') {
        answers.push(JSON.parse(options.body));
        return response({stream_url: answers.length === 1 ? '/voice-answer' : '/text-answer'});
      }
      if (url === '/voice-answer') return timedStream(clock, [
        [2000, 'data: {"type":"answer_delta","text":"  "}\n\n'],
        [3500, 'data: {"type":"answer_delta","text":"先看 offset 提交 [1]"}\n\n'],
        [3600, doneEvent({origin: 'final_chunk_received', ms: {final_chunk_received: 0, asr_final: 400, first_answer_delta: 1900}})],
      ]);
      if (url === '/text-answer') return timedStream(clock, [
        [9000, 'data: {"type":"answer_delta","text":"文本答案"}\n\n'],
        [9100, doneEvent({origin: 'answer_requested', ms: {answer_requested: 0, answer_done: 80}})],
      ]);
      throw new Error(`unexpected fetch ${url}`);
    },
  });
  await app.click();
  const audio = value => ({inputBuffer: {getChannelData: () => new Float32Array(9600).fill(value)}}); // 600 ms
  node.onaudioprocess(audio(0.1));
  node.onaudioprocess(audio(0));
  clock.t = 1000;
  node.onaudioprocess(audio(0)); // 1.2 s post-speech silence: VAD stops here, before partials drain
  clock.t = 1300;
  partial.resolve();
  for (let i = 0; i < 20; i++) await tick();

  assert.deepEqual(answers, [{question: 'Kafka 怎么排查重复消费', language: 'zh', transcript_id: 'timed'}]);
  const meta = app.element('meta').textContent;
  assert.ok(meta.includes('本机：VAD 结束→首个正文 2500 ms'),
    `client latency must run from the VAD stop to the first non-blank delta, got: ${meta}`);
  assert.ok(meta.includes('排空 partial 至发出 final 300 ms'), `partial drain must be separated, got: ${meta}`);
  assert.ok(meta.includes('服务端（自 final_chunk_received，ms）：final_chunk_received 0 · asr_final 400 · first_answer_delta 1900'),
    `server stages must be shown on their own clock, got: ${meta}`);

  app.element('q').value = '文本提问';
  await app.submit();
  assert.deepEqual(answers[1], {question: '文本提问', language: 'zh'}, 'text questions carry no transcript id');
  const textMeta = app.element('meta').textContent;
  assert.ok(!textMeta.includes('本机：'), 'a text answer must not reuse the previous recording timing');
  assert.ok(textMeta.includes('服务端（自 answer_requested，ms）'));
}

const tests = {
  initializationFailureReleasesEverything,
  audioContextFailureReleasesGrantedMicrophone,
  stoppingClosesCallbackBeforeAwaitingQueue,
  stoppingBeforeFirstAudioCallbackCancelsServerSession,
  chosenLanguageFlowsToTextAndTranscriptionAndLocksDuringRecording,
  failedCreationUnlocksLanguage,
  finalTranscriptStartsOneLanguageBoundAnswer,
  cancellationReleasesHardwareAndNeverFinalizes,
  cancellationDiscardsLateTranscriptEvents,
  vadStopsOnlyAfterSpeechAndAccumulatedSilence,
  voiceAnswerReportsClientStopToFirstTextAndServerStages,
};

// PWA_ONLY=<name> runs one regression, e.g. to check it alone against an older revision.
(async () => {
  const only = process.env.PWA_ONLY;
  if (only && !tests[only]) throw new Error(`unknown PWA_ONLY test ${only}`);
  for (const [name, test] of Object.entries(tests)) if (!only || name === only) await test();
  process.stdout.write(`pwa recorder control-flow passed${revision ? ` against ${revision}` : ''}\n`);
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
