// Copy the runtime files the deck needs out of node_modules into vendor/,
// so the presentation runs offline without an install step.
import { cpSync, mkdirSync } from "node:fs";

const copies = [
  ["node_modules/reveal.js/dist/reset.css", "vendor/reveal/reset.css"],
  ["node_modules/reveal.js/dist/reveal.css", "vendor/reveal/reveal.css"],
  ["node_modules/reveal.js/dist/reveal.js", "vendor/reveal/reveal.js"],
  ["node_modules/reveal.js/plugin/notes", "vendor/reveal/plugin/notes"],
  ["node_modules/reveal.js/plugin/zoom", "vendor/reveal/plugin/zoom"],
  ["node_modules/d3/dist/d3.min.js", "vendor/d3.min.js"],
];

mkdirSync("vendor/reveal/plugin", { recursive: true });
for (const [src, dst] of copies) {
  cpSync(src, dst, { recursive: true });
  console.log(`vendored ${dst}`);
}
