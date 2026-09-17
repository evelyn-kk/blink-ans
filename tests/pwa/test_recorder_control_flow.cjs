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

function boot({fetch, getUserMedia, context}) {
  const elements = new Map();
  for (const id of ['f', 'language', 'q', 'go', 'mic', 'transcript', 'status', 'answer', 'sources', 'meta']) {
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
    TextDecoder,
    btoa: value => Buffer.from(value, 'binary').toString('base64'),
    Int16Array, Uint8Array, Math, JSON, Error, Promise, console,
  };
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox, {filename: 'apps/pwa/index.html'});
  elements.get('#language').value = 'zh';
  return {
    click: () => elements.get('#mic').handlers.click(),
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
  assert.deepEqual(answers, [{question: 'How does Kafka compaction work?', language: 'en'}]);
  assert.equal(app.element('q').value, 'How does Kafka compaction work?');
}

(async () => {
  await initializationFailureReleasesEverything();
  await audioContextFailureReleasesGrantedMicrophone();
  await stoppingClosesCallbackBeforeAwaitingQueue();
  await chosenLanguageFlowsToTextAndTranscriptionAndLocksDuringRecording();
  await failedCreationUnlocksLanguage();
  await finalTranscriptStartsOneLanguageBoundAnswer();
  process.stdout.write(`pwa recorder control-flow passed${revision ? ` against ${revision}` : ''}\n`);
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
