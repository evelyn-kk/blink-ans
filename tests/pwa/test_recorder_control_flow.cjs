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
const streamResponse = () => ({ok: true, body: {getReader: () => ({read: async () => ({done: true})})}});

function deferred() {
  let resolve, reject;
  const promise = new Promise((ok, bad) => { resolve = ok; reject = bad; });
  return {promise, resolve, reject};
}

function boot({fetch, getUserMedia, context}) {
  const elements = new Map();
  for (const id of ['f', 'q', 'go', 'mic', 'transcript', 'status', 'answer', 'sources', 'meta']) {
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
  return {click: () => elements.get('#mic').handlers.click(), mic: elements.get('#mic')};
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
  const track = {stop() {}};
  const stream = {getTracks: () => [track]};
  let node;
  const ctx = {sampleRate: 16000, destination: {}, close() {}, createMediaStreamSource: () => ({connect() {}}),
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
  const frozen = chunks.length;
  if (node.onaudioprocess) node.onaudioprocess(audio(0.3));
  assert.equal(chunks.length, frozen, 'post-stop callback must not append a partial');
  partial.resolve();
  await stopping;
  assert.equal(chunks.length, 2, 'only the held final follows the earlier partial');
  assert.equal(chunks[1].final, true);
  assert.equal(chunks.filter(chunk => chunk.final).length, 1, 'final must be unique');
}

(async () => {
  await initializationFailureReleasesEverything();
  await audioContextFailureReleasesGrantedMicrophone();
  await stoppingClosesCallbackBeforeAwaitingQueue();
  process.stdout.write(`pwa recorder control-flow passed${revision ? ` against ${revision}` : ''}\n`);
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
