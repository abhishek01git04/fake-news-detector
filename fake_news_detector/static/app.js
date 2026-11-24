const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let mode = "text";

// Tabs
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    mode = t.dataset.tab;
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== mode));
  })
);
document.querySelectorAll(".ex").forEach((b) => b.addEventListener("click", () => ($("#text").value = b.dataset.text)));

function setStatus(html, isError = false) {
  const s = $("#status");
  s.classList.remove("hidden");
  s.classList.toggle("error", isError);
  s.innerHTML = html;
}

$("#check-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = mode === "url" ? { url: $("#url").value.trim() } : { text: $("#text").value.trim() };
  if (!body.url && !body.text) return setStatus("Please enter some text or a URL.", true);

  $("#submit").disabled = true;
  setStatus('<span class="spinner"></span>Extracting claims, searching the web and verifying evidence… this can take 30–90 seconds.');
  try {
    const res = await fetch("/api/check", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    $("#status").classList.add("hidden");
    renderResult(data);
    loadHistory();
  } catch (err) {
    setStatus(esc(err.message), true);
  } finally {
    $("#submit").disabled = false;
  }
});

function renderResult(r) {
  const pct = r.factuality == null ? null : Math.round(r.factuality * 100);
  const s = r.stats;
  const claims = r.claims
    .map((c) => {
      const ev = c.evidences.filter((e) => e.relationship !== "IRRELEVANT");
      const evHtml = ev.length
        ? ev.map((e) => `<div class="ev"><span class="badge ${e.relationship}">${e.relationship}</span>
            <div>${esc(e.reasoning)}</div><a href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.url)}</a></div>`).join("")
        : '<div class="muted small">No relevant evidence.</div>';
      return `<div class="claim"><div class="claim-head"><div>${esc(c.claim)}</div>
          <span class="badge ${c.status}">${c.status.replace("_", " ")}</span></div>
          ${c.checkworthy ? `<details><summary>Evidence (${ev.length})</summary>${evHtml}</details>` : ""}</div>`;
    })
    .join("");

  const el = $("#result");
  el.classList.remove("hidden");
  el.innerHTML = `
    <h2>Result ${r.demo ? '<span class="badge UNVERIFIABLE">DEMO</span>' : ""}</h2>
    <div class="verdict-row">
      <div class="verdict-big ${r.verdict}">${r.verdict}</div>
      <div class="meter">
        <div>Factuality score: <b>${pct == null ? "n/a" : pct + "%"}</b> · Confidence: <b>${Math.round(r.confidence * 100)}%</b></div>
        <div class="bar"><span style="width:${pct ?? 0}%"></span></div>
        <div class="muted">${esc(r.explanation)}</div>
      </div>
    </div>
    ${r.source_url ? `<p class="muted">Source: <a href="${esc(r.source_url)}" target="_blank" rel="noopener">${esc(r.source_url)}</a></p>` : ""}
    <div class="stats">
      <div class="stat"><b>${s.num_claims}</b>claims</div>
      <div class="stat"><b>${s.num_verified}</b>verified</div>
      <div class="stat"><b style="color:var(--real)">${s.num_supported}</b>supported</div>
      <div class="stat"><b style="color:var(--fake)">${s.num_refuted}</b>refuted</div>
      <div class="stat"><b style="color:var(--mis)">${s.num_disputed}</b>disputed</div>
      <div class="stat"><b>${r.elapsed_seconds}s</b>time</div>
    </div>
    <h2>Claims</h2>${claims || '<p class="muted">No claims were extracted.</p>'}`;
  el.scrollIntoView({ behavior: "smooth" });
}

async function loadHistory() {
  const items = await (await fetch("/api/history")).json();
  const ul = $("#history");
  if (!items.length) return;
  ul.innerHTML = items
    .map((i) => `<li data-id="${i.id}"><span class="badge ${i.verdict}">${i.verdict}</span>
        <span class="muted"> ${esc(i.created_at.replace("T", " "))}</span><span class="snip">${esc(i.snippet)}</span></li>`)
    .join("");
  ul.querySelectorAll("li").forEach((li) =>
    li.addEventListener("click", async () => renderResult(await (await fetch(`/api/history/${li.dataset.id}`)).json()))
  );
}
loadHistory();
