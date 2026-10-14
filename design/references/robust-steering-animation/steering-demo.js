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

  const cachedFields = JSON.parse(document.getElementById("steering-landscapes").textContent).fields;
  const fields = timeline.entries.map(entry => {
    const cached = cachedFields.find(field => field.language === entry.language);
    const node = document.createElementNS("http://www.w3.org/2000/svg", "g");
    node.replaceChildren(...cached.paths.map(attributes => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      Object.entries(attributes).forEach(([key, value]) => path.setAttribute(key, value));
      return path;
    }));
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
      const route = cachedFields.find(field => field.language === entry.language).route;
      if (route) {
        find("route-line").setAttribute("d", route.d);
        ["start", "end"].forEach(name => {
          find("route-" + name).setAttribute("cx", route[name][0]);
          find("route-" + name).setAttribute("cy", route[name][1]);
        });
        find("route-head").setAttribute("transform", "translate(" + route.end.join(" ") + ") rotate(" + route.angle + ")");
      }
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
    const steering = snapshot.phase === "steering" || snapshot.phase === "steered";
    find("route").setAttribute("opacity", steering ? "1" : "0");
    find("route-line").setAttribute("stroke-dashoffset", 1 - snapshot.corruption);
    find("route-head").setAttribute("opacity", snapshot.corruption === 1 ? "1" : "0");
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
      if (phase === "baseline-hold") description.textContent = entry.label + " question: " + entry.question + " Answer: " + entry.baseline;
      if (phase === "steered") description.textContent = "Steered answer: " + entry.steered;
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
