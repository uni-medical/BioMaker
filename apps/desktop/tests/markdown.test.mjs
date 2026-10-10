import assert from 'node:assert/strict';
import { test } from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ReactMarkdown from 'react-markdown';
import { markdownOptions } from '../src/markdown.ts';
const render = text => renderToStaticMarkup(React.createElement(ReactMarkdown, markdownOptions, text));

test('report tables render semantic headers and cells, including Unicode metrics', () => {
  const html = render('| 候选 | AUROC |\n| --- | ---: |\n| 最佳方案 | 0.927266 |');
  assert.match(html, /<table>/); assert.match(html, /<th>候选<\/th>/);
  assert.match(html, /<td>最佳方案<\/td>/); assert.match(html, /<td style="text-align:right">0.927266<\/td>/);
});
test('GFM keeps raw HTML and dangerous URL schemes out of rendered reports', () => {
  const html = render('<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\n[unsafe](javascript:alert%281%29)\n\n![unsafe](data:text/html,payload)\n\n[docs](https://example.org)');
  assert.doesNotMatch(html, /<script|onerror|javascript:|data:text\/html/);
  assert.match(html, /href="https:\/\/example.org"/);
});
