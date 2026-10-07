// Deck runtime: reveal.js set-up, the footer progress tracker, and the bridge
// that forwards fragment progress to embedded figure iframes.

const PARTS = [
  ["context", "Context"],
  ["data", "Data"],
  ["twin", "Digital twin"],
  ["evidence", "Experiment 3"],
  ["outlook", "Outlook"],
];

function buildFooter() {
  const footer = document.createElement("div");
  footer.className = "deck-footer";
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = "Geospatial Systems CDT · Newcastle University · DSTL Review · October 2026";
  footer.appendChild(who);
  for (const [key, label] of PARTS) {
    const p = document.createElement("span");
    p.className = "part";
    p.dataset.part = key;
    p.textContent = label;
    footer.appendChild(p);
  }
  document.querySelector(".reveal").appendChild(footer);
}

function updateFooter(slide) {
  const part = slide.dataset.part || "";
  const idx = PARTS.findIndex(([k]) => k === part);
  document.querySelectorAll(".deck-footer .part").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
    el.classList.toggle("done", idx >= 0 && i < idx);
  });
  const root = document.documentElement;
  root.classList.toggle("hide-chrome", slide.dataset.chrome === "none");
  root.classList.toggle("dark-chrome", slide.dataset.chrome === "dark");
}

// Number of fragments currently shown on a slide (0 when none).
function fragmentStep(slide) {
  return slide.querySelectorAll(".fragment.visible").length;
}

function postStep(slide) {
  const step = fragmentStep(slide);
  slide.querySelectorAll("iframe.fig-frame").forEach((frame) => {
    if (frame.contentWindow) frame.contentWindow.postMessage({ type: "reveal-step", step }, "*");
  });
}

// A lazily loaded figure announces itself; reply with the current step.
window.addEventListener("message", (e) => {
  if (!e.data || e.data.type !== "figure-ready") return;
  const slide = Reveal.getCurrentSlide();
  slide.querySelectorAll("iframe.fig-frame").forEach((frame) => {
    if (frame.contentWindow === e.source) {
      frame.contentWindow.postMessage({ type: "reveal-step", step: fragmentStep(slide) }, "*");
    }
  });
});

Reveal.initialize({
  width: 1920,
  height: 1080,
  margin: 0,
  minScale: 0.2,
  maxScale: 2.0,
  center: false,
  hash: true,
  history: false,
  slideNumber: "c/t",
  controls: false,
  progress: true,
  transition: "fade",
  transitionSpeed: "fast",
  backgroundTransition: "fade",
  preloadIframes: true,
  viewDistance: 2,
  plugins: [RevealNotes, RevealZoom],
}).then(() => {
  buildFooter();
  updateFooter(Reveal.getCurrentSlide());
  postStep(Reveal.getCurrentSlide());
});

Reveal.on("slidechanged", (e) => {
  updateFooter(e.currentSlide);
  postStep(e.currentSlide);
});
Reveal.on("fragmentshown", () => postStep(Reveal.getCurrentSlide()));
Reveal.on("fragmenthidden", () => postStep(Reveal.getCurrentSlide()));
