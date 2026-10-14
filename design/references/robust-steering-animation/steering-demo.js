"use strict";

(() => {
  const box = document.querySelector(".steering-demo");
  if (!box) return;
  const examples = JSON.parse(document.getElementById("steering-examples").textContent);
  const timeline = SteeringTimeline.create(examples);
  const find = (name) => box.querySelector(".steering-demo-" + name);
  const question = find("question-text");
  const response = find("answer-text");
  const description = find("description");
  const pauseButton = find("pause");
  const scene = find("scene");
  const sceneFrame = find("scene-frame");
  const state = find("state");
  const direction = find("direction");
  const directionLine = find("direction-line");
  const directionHead = find("direction-head");
  const directionLabels = find("direction-labels");
  const shiftMask = find("shift-mask");
  const baseMask = find("base-mask");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const glyphs = Array.from("▓▒░<>/\\#¿!?");
  let responseChars;
  let fieldPair = '';
  let elapsed = 0;
  let previousTime = null;
  let frame = null;
  let inView = false;
  let paused = reducedMotion.matches;
  let motionAllowed = !reducedMotion.matches;
  let started = false;
  let flightEnd;
  let sceneScale = 1;
  const clamp = (x) => Math.max(0, Math.min(1, x));
  const ease = (x) => x * x * (3 - 2 * x);

  function makeStateContours(node, seed, components, color) {
    // Seeded Gaussian-mixture samples and a Gaussian KDE, not hand-drawn ellipses.
    // Synthetic illustration only: these are not recorded model activations.
    const random = () => {
      seed |= 0;
      seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
    const totalWeight = components.reduce((sum, c) => sum + c.weight, 0);
    const samples = Array.from({ length: 600 }, () => {
      let draw = random() * totalWeight;
      const c = components.find((component) => (draw -= component.weight) < 0) || components[components.length - 1];
      const radius = Math.sqrt(-2 * Math.log(Math.max(random(), 1e-10)));
      const theta = 2 * Math.PI * random();
      const a = radius * Math.cos(theta);
      const b = radius * Math.sin(theta);
      return [c.x + c.sx * a, c.y + c.sy * (c.rho * a + Math.sqrt(1 - c.rho * c.rho) * b)];
    });
    // Evaluate beyond the visible square, so contours can cross its edges cleanly.
    const n = 128;
    const density = new Float64Array(n * n);
    const bandwidth = .45;
    let peak = 0;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const px = -6 + x * 12 / (n - 1);
        const py = 6 - y * 12 / (n - 1);
        let value = 0;
        for (const [sx, sy] of samples) {
          const distance = ((px - sx) ** 2 + (py - sy) ** 2) / (2 * bandwidth * bandwidth);
          if (distance < 18) value += Math.exp(-distance);
        }
        density[y * n + x] = value;
        peak = Math.max(peak, value);
      }
    }
    const cases = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 0], [1, 2]], [[0, 2]], [[3, 2]], [[2, 3]], [[0, 2]], [[0, 1], [2, 3]], [[1, 2]], [[1, 3]], [[0, 1]], [[3, 0]], []];
    const paths = [];
    for (const fraction of [.04, .08, .14, .23, .34, .47, .61, .76, .9]) {
      const threshold = peak * fraction;
      const vertices = new Map();
      const adjacency = new Map();
      const segments = [];
      for (let y = 0; y < n - 1; y++) {
        for (let x = 0; x < n - 1; x++) {
          const values = [density[y * n + x], density[y * n + x + 1], density[(y + 1) * n + x + 1], density[(y + 1) * n + x]];
          const mask = values.reduce((code, v, i) => code | (v >= threshold ? 1 << i : 0), 0);
          if (!cases[mask].length) continue;
          const corners = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
          const keys = ["h:" + x + ":" + y, "v:" + (x + 1) + ":" + y, "h:" + x + ":" + (y + 1), "v:" + x + ":" + y];
          const vertex = (edge) => {
            const key = keys[edge];
            if (!vertices.has(key)) {
              const next = (edge + 1) % 4;
              const t = (threshold - values[edge]) / (values[next] - values[edge]);
              const [ax, ay] = corners[edge];
              const [bx, by] = corners[next];
              vertices.set(key, [-70 + (ax + (bx - ax) * t) * 480 / (n - 1), -82 + (ay + (by - ay) * t) * 480 / (n - 1)]);
            }
            return key;
          };
          for (const [a, b] of cases[mask]) {
            const pair = [vertex(a), vertex(b)];
            const index = segments.push(pair) - 1;
            for (const key of pair) {
              if (!adjacency.has(key)) adjacency.set(key, []);
              adjacency.get(key).push(index);
            }
          }
        }
      }
      const used = new Set();
      let d = "";
      segments.forEach(([start, end], index) => {
        if (used.has(index)) return;
        const points = [vertices.get(start), vertices.get(end)];
        used.add(index);
        let key = end;
        while (key !== start) {
          const nextIndex = adjacency.get(key).find((i) => !used.has(i));
          if (nextIndex === undefined) break;
          used.add(nextIndex);
          const pair = segments[nextIndex];
          key = pair[0] === key ? pair[1] : pair[0];
          points.push(vertices.get(key));
        }
        // Closed rings get a very light density fill; open boundary contours stay unfilled.
        if (key !== start) return;
        d += "M" + points.map(([x, y]) => x.toFixed(2) + "," + y.toFixed(2)).join("L") + "Z";
      });
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", color);
      path.setAttribute("fill-opacity", ".055");
      path.setAttribute("fill-rule", "evenodd");
      path.setAttribute("stroke", color);
      path.setAttribute("stroke-opacity", ".6");
      path.setAttribute("stroke-width", "1.1");
      path.setAttribute("stroke-linejoin", "round");
      paths.push(path);
    }
    node.replaceChildren(...paths);
    node.dataset.sampleCount = samples.length;
    node.dataset.method = "Seeded Gaussian-mixture sampling; Gaussian KDE; marching-squares contours";
  }

  // Compute once, then reuse exactly the same fields for every replay.
  // The multi-modal landscapes are independent of the two language-direction points.
  const englishComponents = [
    { weight: .25, x: 0, y: 0, sx: 2.6, sy: 2.6, rho: -.1 },
    { weight: .13, x: -2.4, y: 1.5, sx: .75, sy: .9, rho: -.3 },
    { weight: .1, x: -.45, y: 2.2, sx: 1.0, sy: .55, rho: -.2 },
    { weight: .13, x: 2.0, y: 1.5, sx: .8, sy: .8, rho: .2 },
    { weight: .12, x: -1.75, y: -.7, sx: .9, sy: .65, rho: .1 },
    { weight: .1, x: .7, y: .1, sx: .75, sy: .9, rho: -.4 },
    { weight: .08, x: 2.5, y: -1.5, sx: .7, sy: .8, rho: -.2 },
    { weight: .09, x: -.4, y: -2.4, sx: 1.15, sy: .55, rho: .1 },
  ];
  const shiftedComponents = [
    { weight: .3, x: 0, y: 0, sx: 2.9, sy: 2.9, rho: .08 },
    { weight: .13, x: -1.7, y: 2.1, sx: .85, sy: 1.0, rho: .5 },
    { weight: .16, x: 1.2, y: -1.6, sx: 1.1, sy: .8, rho: .5 },
    { weight: .1, x: .7, y: 1.3, sx: .8, sy: .55, rho: -.45 },
    { weight: .07, x: -1.6, y: -1.7, sx: .9, sy: .5, rho: -.2 },
    { weight: .06, x: 2.8, y: 1.8, sx: .55, sy: .9, rho: -.3 },
    { weight: .05, x: -3.1, y: -.1, sx: .7, sy: .85, rho: .2 },
    { weight: .04, x: -.2, y: -.2, sx: .4, sy: .4, rho: -.1 },
    { weight: .035, x: -2.9, y: 2.5, sx: .4, sy: .4, rho: .2 },
    { weight: .03, x: 2.1, y: -3, sx: .7, sy: .45, rho: .15 },
    { weight: .025, x: 2.7, y: -.1, sx: .45, sy: .6, rho: -.2 },
  ];
  const fields = timeline.entries.map((entry, index) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const english = entry.language === "en";
    makeStateContours(node, 2027 + index, english ? englishComponents : shiftedComponents, english ? "#919191" : "#8B1E1E");
    return node;
  });

  function characterNodes(node, count) {
    const nodes = Array.from({ length: count }, () => {
      const span = document.createElement("span");
      span.className = "steering-demo-char";
      return span;
    });
    node.replaceChildren(...nodes);
    return nodes;
  }

  function measureFlight() {
    const bounds = box.getBoundingClientRect();
    box.style.setProperty("--demo-svg-scale-inverse", 340 / find("state-map").clientWidth);
    sceneScale = parseFloat(getComputedStyle(box).getPropertyValue("--demo-chat-scale")) * find("robot").clientWidth / 58;
    const model = find("robot").getBoundingClientRect();
    find("stimulation").style.transform = "translate(" + (model.x + model.width / 2 - bounds.x - 45 * sceneScale) + "px," + (model.y - bounds.y - 72 * sceneScale) + "px) scale(" + sceneScale + ")";
    const to = find("bolt").getBoundingClientRect();
    flightEnd = new DOMPoint(to.x + to.width / 2, to.y + to.height / 2)
      .matrixTransform(find("state-map").getScreenCTM().inverse());
  }

  function reserveResponseHeight() {
    const questionSample = question.cloneNode(false);
    questionSample.removeAttribute("data-typing");
    questionSample.style.cssText = "position:absolute;visibility:hidden;pointer-events:none;min-height:0;height:auto;width:" + question.clientWidth + "px";
    question.parentElement.append(questionSample);
    const questionHeight = Math.max(...examples.map(({ question: text }) => {
      questionSample.textContent = text;
      return questionSample.offsetHeight;
    }));
    question.style.height = questionHeight + "px";
    questionSample.remove();
    // Reserve every mixed/glitched response too, not only the two complete answers.
    // Inline-block glitch glyphs can add a line; they must not resize the scene.
    const sample = response.cloneNode(false);
    sample.removeAttribute("data-typing");
    sample.style.cssText = "position:absolute;visibility:hidden;pointer-events:none;min-height:0;height:auto;width:" + response.clientWidth + "px";
    response.parentElement.append(sample);
    let height = Math.max(...examples.flatMap(entry => [entry.baseline, entry.steered || entry.baseline]).map((text) => {
      sample.textContent = text;
      return sample.offsetHeight;
    }));
    const probe = document.createElement("span");
    probe.style.display = "inline-block";
    sample.replaceChildren(probe);
    let widestGlyph = glyphs[0], widestWidth = 0;
    glyphs.forEach((glyph) => {
      probe.textContent = glyph;
      if (probe.offsetWidth > widestWidth) {
        widestWidth = probe.offsetWidth;
        widestGlyph = glyph;
      }
    });
    const mixed = characterNodes(sample, responseChars.length);
    for (const entry of timeline.entries) {
      if (!entry.steered) continue;
      for (let front = 0; front <= mixed.length + 5; front++) {
        mixed.forEach((span, i) => {
          const changing = i <= front && i > front - 5;
          span.classList.add("is-visible");
          span.classList.toggle("is-glitch", changing);
          span.textContent = changing ? widestGlyph : i <= front - 5 ? entry.units.steered[i] || "" : entry.units.baseline[i] || "";
        });
        height = Math.max(height, sample.offsetHeight);
      }
    }
    response.style.setProperty("--demo-response-height", height + "px");
    sample.remove();
    const scale = parseFloat(getComputedStyle(box).getPropertyValue("--demo-chat-scale"));
    sceneFrame.style.height = scene.offsetHeight * scale + "px";
    measureFlight();
  }

  function prepare() {
    responseChars = characterNodes(response, Math.max(...timeline.entries.flatMap(entry => [entry.units.baseline.length, entry.units.steered.length])));
    elapsed = 0;
    previousTime = null;
    render();
    reserveResponseHeight();
  }

  function renderState(snapshot) {
    const { entry, previous, index, first, time } = snapshot;
    const previousIndex = first ? index : (index + fields.length - 1) % fields.length;
    const pair = previousIndex + ":" + index;
    if (pair !== fieldPair) {
      [find("field-english"), find("field-chinese")].forEach((node, i) => {
        node.replaceChildren(...[...fields[i ? index : previousIndex].children].map(child => child.cloneNode(true)));
      });
      const labels = find("language").querySelectorAll("span");
      [first ? entry : previous, entry].forEach((item, i) => {
        labels[i].textContent = item.label;
        labels[i].style.color = item.language === "en" ? "#111" : "#8B1E1E";
      });
      fieldPair = pair;
    }
    const progress = (start, span) => clamp((time - start) / span);
    const english = entry.language === "en";
    const offset = first ? 0 : 2200;
    box.dataset.stage = snapshot.swipe < 1 && !first ? "shift" : english ? "direction" : "conversation";
    state.style.opacity = 1;
    const points = english ? progress(offset + 400, 300) : 0;
    const drawn = english ? progress(offset + 1000, 850) : 0;
    const stored = ease(progress(offset + 2800, 1100));
    direction.style.opacity = points * (1 - progress(offset + 3700, 200));
    directionLine.style.opacity = drawn;
    directionHead.style.opacity = clamp(drawn * 4 - 3);
    directionLabels.style.opacity = points * (1 - progress(offset + 2800, 300));
    box.style.setProperty("--demo-direction", drawn);
    const scale = 1 - stored * .94;
    // Carry the direction straight to the model's lightning, with no controller stop.
    const target = flightEnd || { x: 169, y: 160 };
    direction.setAttribute("transform", "translate(" + (169 * (1 - scale) + (target.x - 169) * stored) + " " + (160 * (1 - scale) + (target.y - 160) * stored) + ") scale(" + scale + ")");
    const swipe = ease(snapshot.swipe);
    box.style.setProperty("--demo-swipe", swipe);
    shiftMask.setAttribute("width", swipe * 280);
    baseMask.setAttribute("x", 30 + swipe * 280);
    baseMask.setAttribute("width", (1 - swipe) * 280);
    find("stimulation").style.opacity = 1;
    scene.style.opacity = 1;
    scene.style.clipPath = "none";
  }

  function render() {
    const snapshot = timeline.sample(elapsed);
    renderState(snapshot);
    const { entry, phase } = snapshot;
    box.dataset.language = entry.language;
    if (box.dataset.phase !== phase) {
      box.dataset.phase = phase;
      if (phase === "baseline-hold") description.textContent = "Recorded " + entry.label + " question: " + entry.question + " Answer excerpt: " + entry.baseline;
      if (phase === "steered") description.textContent = "Recorded steered answer excerpt: " + entry.steered;
    }
    if (question.textContent !== snapshot.question) question.textContent = snapshot.question;
    question.lang = snapshot.questionLanguage;
    response.lang = snapshot.answerLanguage;
    question.dataset.typing = String(snapshot.questionTyping);
    response.dataset.typing = String(snapshot.answerTyping);
    const answerUnits = timeline.units(snapshot.answer);
    const front = snapshot.corruption * (Math.max(entry.units.baseline.length, entry.units.steered.length) + 5);
    responseChars.forEach((span, i) => {
      const changing = phase === "steering" && i <= front && i > front - 5;
      let char = answerUnits[i] || "";
      if (phase === "steering") char = i <= front - 5 ? entry.units.steered[i] || "" : entry.units.baseline[i] || "";
      if (changing) char = glyphs[(i * 7 + Math.floor(elapsed / 65)) % glyphs.length];
      if (span.textContent !== char) span.textContent = char;
      span.classList.add("is-visible");
      span.classList.toggle("is-glitch", changing);
    });
  }

  function stopFrame() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    previousTime = null;
  }

  function tick(now) {
    frame = null;
    if (paused || !inView || document.hidden || !motionAllowed) return;
    if (previousTime !== null) elapsed += Math.min(now - previousTime, 100);
    previousTime = now;
    render();
    frame = requestAnimationFrame(tick);
  }

  function resume() {
    if (!paused && inView && !document.hidden && motionAllowed && frame === null) {
      previousTime = null;
      frame = requestAnimationFrame(tick);
    }
  }

  function replay() {
    stopFrame();
    elapsed = 0;
    started = true;
    box.dataset.paused = String(paused);
    pauseButton.textContent = paused ? "Play" : "Pause";
    pauseButton.setAttribute("aria-label", paused ? "Play animation" : "Pause animation");
    render();
    resume();
  }

  pauseButton.addEventListener("click", () => {
    paused = !paused;
    if (!paused) motionAllowed = true;
    box.dataset.paused = String(paused);
    pauseButton.textContent = paused ? "Play" : "Pause";
    pauseButton.setAttribute("aria-label", paused ? "Play animation" : "Pause animation");
    if (paused) stopFrame(); else resume();
  });
  find("replay").addEventListener("click", replay);
  new IntersectionObserver(([entry]) => {
    inView = entry.isIntersecting;
    if (inView && !started) replay();
    else if (inView) resume();
    else stopFrame();
  }, { threshold: .25 }).observe(box);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopFrame(); else resume();
  });
  reducedMotion.addEventListener("change", () => {
    motionAllowed = !reducedMotion.matches;
    paused = reducedMotion.matches;
    replay();
  });
  new ResizeObserver(reserveResponseHeight).observe(sceneFrame);
  box.steeringDemo = { replay };
  box.dataset.paused = String(paused);
  prepare();
})();
