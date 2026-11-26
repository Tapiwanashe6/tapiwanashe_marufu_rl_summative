# /// script
# requires-python = ">=3.11"
# dependencies = ["python-docx>=1.1"]
# ///
"""Fill the official ALU report template (.docx) with the report content.

    uv run report/fill_template.py TEMPLATE.docx OUT.docx [--video URL]

Placeholders are replaced in place; every added paragraph/table is cloned from an element
of the template itself, so fonts, sizes, borders and spacing stay exactly as the template sets them.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "assets" / "figures"
W = qn


# ------------------------------------------------------------------ low-level helpers
def _rpr(bold=False, italic=False, size=20, font="Times New Roman"):
    rpr = Document().element.makeelement(W("w:rPr"), {})
    f = rpr.makeelement(W("w:rFonts"), {W(f"w:{k}"): font for k in ("ascii", "cs", "eastAsia", "hAnsi")})
    rpr.append(f)
    if bold:
        rpr.append(rpr.makeelement(W("w:b"), {W("w:val"): "1"})); rpr.append(rpr.makeelement(W("w:bCs"), {W("w:val"): "1"}))
    if italic:
        rpr.append(rpr.makeelement(W("w:i"), {W("w:val"): "1"})); rpr.append(rpr.makeelement(W("w:iCs"), {W("w:val"): "1"}))
    rpr.append(rpr.makeelement(W("w:sz"), {W("w:val"): str(size)})); rpr.append(rpr.makeelement(W("w:szCs"), {W("w:val"): str(size)}))
    return rpr


INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)")


def _set_runs(p_el, text: str, bold=False, italic=False, size=20, font="Times New Roman"):
    """Remove all runs from paragraph element p_el and write `text` (supports **bold** and *italic*)."""
    for child in list(p_el):
        if child.tag != W("w:pPr"):
            p_el.remove(child)
    for part in INLINE.split(text):
        if not part:
            continue
        b, i, t = bold, italic, part
        if part.startswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            i, t = True, part[1:-1]
        r = p_el.makeelement(W("w:r"), {}); r.append(_rpr(b, i, size, font))
        te = r.makeelement(W("w:t"), {}); te.text = t; te.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(te); p_el.append(r)


def _plain_ppr(p_el, before=0, after=120, center=False, keep_next=False):
    """Make p_el's pPr a plain body paragraph (no list numbering, no italics), set spacing."""
    ppr = p_el.find(W("w:pPr"))
    if ppr is None:
        ppr = p_el.makeelement(W("w:pPr"), {}); p_el.insert(0, ppr)
    for tag in ("w:numPr", "w:ind", "w:jc", "w:pStyle", "w:keepNext", "w:spacing"):
        for e in ppr.findall(W(tag)):
            ppr.remove(e)
    rpr = ppr.find(W("w:rPr"))
    if rpr is not None:
        for tag in ("w:i", "w:iCs", "w:b", "w:bCs"):
            for e in rpr.findall(W(tag)):
                rpr.remove(e)
    sp = ppr.makeelement(W("w:spacing"), {W("w:before"): str(before), W("w:after"): str(after), W("w:lineRule"): "auto"})
    ind = ppr.makeelement(W("w:ind"), {W("w:left"): "0", W("w:firstLine"): "0"})
    extra = [sp, ind]
    if keep_next:
        extra.insert(0, ppr.makeelement(W("w:keepNext"), {W("w:val"): "1"}))
    if center:
        extra.append(ppr.makeelement(W("w:jc"), {W("w:val"): "center"}))
    else:
        extra.append(ppr.makeelement(W("w:jc"), {W("w:val"): "both"}))
    pos = 0
    for e in extra:
        ppr.insert(pos, e); pos += 1
    # keep pPr children before rPr in schema order (rPr must be last)
    if rpr is not None:
        ppr.remove(rpr); ppr.append(rpr)


class Filler:
    def __init__(self, path: Path):
        self.doc = Document(str(path))
        self.P = list(self.doc.paragraphs)          # original paragraphs, indexed as in the template
        self.T = list(self.doc.tables)
        self.proto_p = copy.deepcopy(self.P[35]._p)  # a plain results-section body paragraph
        self.proto_tbl = copy.deepcopy(self.T[0]._tbl)   # bold-header 5-col table (Observation table)

    # paragraphs ------------------------------------------------------------
    def para_after(self, anchor, text, *, before=0, after=120, center=False, size=20, bold=False, italic=False, keep_next=False, font="Times New Roman"):
        p = copy.deepcopy(self.proto_p)
        _plain_ppr(p, before, after, center, keep_next)
        _set_runs(p, text, bold=bold, italic=italic, size=size, font=font)
        anchor.addnext(p)
        return p

    def replace(self, idx, text, *, before=None, after=None):
        p = self.P[idx]._p
        _set_runs(p, text)
        ppr = p.find(W("w:pPr"))
        rpr = ppr.find(W("w:rPr")) if ppr is not None else None
        if rpr is not None:
            for tag in ("w:i", "w:iCs"):
                for e in rpr.findall(W(tag)):
                    rpr.remove(e)
        if before is not None or after is not None:
            sp = ppr.find(W("w:spacing"))
            if sp is None:
                sp = ppr.makeelement(W("w:spacing"), {}); ppr.insert(0, sp)
            if before is not None: sp.set(W("w:before"), str(before))
            if after is not None: sp.set(W("w:after"), str(after))
        jc = ppr.find(W("w:jc"))
        if jc is None:
            jc = ppr.makeelement(W("w:jc"), {}); rp = ppr.find(W("w:rPr"))
            (rp.addprevious(jc) if rp is not None else ppr.append(jc))
        jc.set(W("w:val"), "both")
        return p

    def paras(self, anchor, texts, **kw):
        for t in texts:
            anchor = self.para_after(anchor, t, **kw)
        return anchor

    def clone_label(self, src_idx, anchor, text):
        p = copy.deepcopy(self.P[src_idx]._p)
        rpr = p.find(W("w:r")).find(W("w:rPr"))
        for child in list(p):
            if child.tag != W("w:pPr"):
                p.remove(child)
        r = p.makeelement(W("w:r"), {}); r.append(copy.deepcopy(rpr))
        te = r.makeelement(W("w:t"), {}); te.text = text; te.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(te); p.append(r)
        anchor.addnext(p)
        return p

    def figure(self, anchor, path: Path, width_cm: float, caption: str):
        p = self.para_after(anchor, "", before=60, after=0, center=True, keep_next=True)
        Paragraph(p, self.doc._body).add_run().add_picture(str(path), width=Cm(width_cm))
        return self.para_after(p, caption, after=160, center=True, size=18, italic=True)

    # tables ----------------------------------------------------------------
    def fill_table(self, tbl, header, rows, widths, *, bold_rows=(), size=None):
        """Rewrite a template table: same borders/margins/fonts, new column count and rows."""
        assert abs(sum(widths) - 9029) < 2, sum(widths)
        trs = tbl.findall(W("w:tr"))
        hdr_tr, data_tr = trs[0], trs[1]
        hdr_size = size or self._cell_size(hdr_tr)
        data_size = size or self._cell_size(data_tr)
        hdr_bold = self._cell_bold(hdr_tr)
        for tr in trs:
            tbl.remove(tr)
        grid = tbl.find(W("w:tblGrid"))
        for g in list(grid):
            grid.remove(g)
        for w in widths:
            grid.append(grid.makeelement(W("w:gridCol"), {W("w:w"): str(w)}))
        tbl.append(self._row(hdr_tr, header, widths, hdr_bold, hdr_size, header=True))
        for k, r in enumerate(rows):
            tbl.append(self._row(data_tr, r, widths, k in bold_rows, data_size))

    def new_table(self, anchor, header, rows, widths, **kw):
        t = copy.deepcopy(self.proto_tbl)
        anchor.addnext(t)
        self.fill_table(t, header, rows, widths, **kw)
        return t

    @staticmethod
    def _cell_size(tr):
        sz = tr.find(".//" + W("w:r") + "/" + W("w:rPr") + "/" + W("w:sz"))
        return int(sz.get(W("w:val"))) if sz is not None else 20

    @staticmethod
    def _cell_bold(tr):
        b = tr.find(".//" + W("w:r") + "/" + W("w:rPr") + "/" + W("w:b"))
        return b is not None and b.get(W("w:val"), "1") in ("1", "true")

    def _row(self, proto_tr, cells, widths, bold, size, header=False):
        tr = copy.deepcopy(proto_tr)
        tcs = tr.findall(W("w:tc"))
        proto_tc = tcs[0]
        for tc in tcs:
            tr.remove(tc)
        trpr = tr.find(W("w:trPr"))
        if trpr is not None:                       # the template's empty placeholder row has a 1.4 cm minimum height
            for e in trpr.findall(W("w:trHeight")):
                trpr.remove(e)
            for e in trpr.findall(W("w:cantSplit")):
                e.set(W("w:val"), "1")
        if header:
            th = trpr.find(W("w:tblHeader")) if trpr is not None else None
            if th is not None:
                th.set(W("w:val"), "1")
        for text, w in zip(cells, widths):
            tc = copy.deepcopy(proto_tc)
            tcpr = tc.find(W("w:tcPr"))
            for e in tcpr.findall(W("w:tcW")):
                tcpr.remove(e)
            tcpr.insert(0, tcpr.makeelement(W("w:tcW"), {W("w:w"): str(w), W("w:type"): "dxa"}))
            mar = tcpr.find(W("w:tcMar"))
            if mar is not None:                    # template: 100 dxa all round; keep sides, tighten top/bottom
                for side in ("w:top", "w:bottom"):
                    e = mar.find(W(side))
                    if e is not None:
                        e.set(W("w:w"), "30")
            ps = tc.findall(W("w:p"))
            for extra in ps[1:]:
                tc.remove(extra)
            p = ps[0]
            ppr = p.find(W("w:pPr"))
            if ppr is not None:
                for e in ppr.findall(W("w:spacing")):
                    ppr.remove(e)
                ppr.insert(0, ppr.makeelement(W("w:spacing"), {W("w:before"): "0", W("w:after"): "0", W("w:line"): "240", W("w:lineRule"): "auto"}))
            _set_runs(p, text, bold=bold, size=size)
            tr.append(tc)
        return tr

    def save(self, out: Path):
        self.doc.save(str(out))


# ------------------------------------------------------------------ content
# A real step record from visualization/episodes/dqn_followup_dqn_2m_wide_seed42.json (step 2), fields trimmed.
STEP_JSON = ('{"step":2,"action_name":"fly:Hospital South","reward":9.2095,"reward_breakdown":{"energy":-0.7905,'
             '"delivery_on_time":10.0},"t_start":2.0,"t_end":14.88,"events":[{"t":2.0,"type":"takeoff","destination":3,'
             '"weather":"storm","expected_energy":38.06},{"t":10.0,"type":"weather","weather":"windy"},{"t":13.88,'
             '"type":"deliver","package":"blood","on_time":true}],"state":{"battery":60.474,"location":3,"payload":[10]}}')

REFERENCES = [
    "[1] M. Towers et al., Gymnasium, arXiv:2407.17032, 2024.  [2] A. Bou et al., TorchRL, ICLR 2024.  "
    "[3] three.js docs, r169, threejs.org/docs.  [4] V. Mnih et al., Human-level control through deep RL, Nature 518, "
    "2015.  [5] H. van Hasselt et al., Double Q-learning, AAAI 2016.  [6] R. J. Williams, REINFORCE, Machine Learning 8, "
    "1992.  [7] V. Mnih et al., Asynchronous methods (A3C/A2C), ICML 2016.  [8] J. Schulman et al., GAE, ICLR 2016.  "
    "[9] J. Schulman et al., Proximal policy optimization, arXiv:1707.06347, 2017.",
]


def _link_after(doc, t, url: str) -> None:
    """Insert a clickable external hyperlink showing `url` right after the run holding w:t `t`."""
    run = t.getparent()
    rid = doc.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                             is_external=True)
    link = run.makeelement(W("w:hyperlink"), {qn("r:id"): rid})
    new = copy.deepcopy(run)
    for child in list(new):
        if child.tag != W("w:rPr"):
            new.remove(child)
    rpr = new.find(W("w:rPr"))
    if rpr is None:
        rpr = new.makeelement(W("w:rPr"), {})
        new.insert(0, rpr)
    for tag in ("w:b", "w:color", "w:u"):
        for e in rpr.findall(W(tag)):
            rpr.remove(e)
    rpr.append(rpr.makeelement(W("w:color"), {W("w:val"): "0563C1"}))
    rpr.append(rpr.makeelement(W("w:u"), {W("w:val"): "single"}))
    txt = new.makeelement(W("w:t"), {})
    txt.text = url
    new.append(txt)
    link.append(new)
    # if the run has content after this w:t (e.g. a <w:br/>), split so the link sits before it
    after = list(run)[list(run).index(t) + 1:]
    run.addnext(link)
    if after:
        tail = copy.deepcopy(run)
        for child in list(tail):
            if child.tag != W("w:rPr"):
                tail.remove(child)
        for child in after:
            tail.append(child)
        link.addnext(tail)


def fill(template: Path, out: Path, video: str | None) -> None:
    f = Filler(template)
    P = f.P

    # --- header block
    p1 = P[1]._p
    runs = p1.findall(W("w:r"))
    runs[1].find(W("w:t")).text = " Tapiwanashe Gift Marufu"
    p2 = P[2]._p
    ts = p2.findall(".//" + W("w:t"))
    ts[1].text = " "                                          # the line break is a separate <w:br/>
    ts[3].text = " "
    if video:
        _link_after(f.doc, ts[1], video)
    else:
        ts[1].text = " [link to be added]"
    _link_after(f.doc, ts[3], "https://github.com/Tapiwanashe6/tapiwanashe_marufu_rl_summative")

    # --- Project overview
    f.replace(4, "MediFlight AI trains an autonomous medical-delivery drone to dispatch blood, vaccines, emergency kits and medicines from one distribution centre to 11 hospitals, pharmacies and homes during a 4-hour shift. Demand exceeds what one drone can serve, so it must decide what to load, where to fly and when to charge under deadlines, cold-chain limits and changing weather. I built a custom Gymnasium [1] environment, trained DQN, REINFORCE, PPO and A2C with TorchRL [2], and compared them with dispatch heuristics on held-out scenarios.")

    # --- Agent(s)
    ag = f.replace(7, "One autonomous VTOL medical drone based at the Medical Distribution Center (node 0, with a charging pad). It can load packages at the depot, fly between any of the 12 nodes (lowering a package on arrival), return, charge or hold; 100% battery, 2.5 kg / 3-package payload, 60 km/h calm-air speed. **Why RL:** sequential resource allocation under stochastic demand and weather, with delayed rewards and irreversible failures (crash, spoilage, expiry); good decisions depend on the future, which one-step ranking rules cannot capture, and the results test this against two such rules.")
    f.para_after(ag, "**How transitions are produced.** Each action advances a simulated clock; weather ticks and requests inside it are processed in time order. Leg energy, in battery percentage points (capacity 100), is E = d·(1.6 + 0.5·m)·w + 1 for d km and m kg, w = 1.0 / 1.3 / 1.8 (calm / windy / storm); flight time is d / (1 km/min × 1.0 / 0.85 / 0.7) + 1 min, both fixed at take-off. Gust noise multiplies E by a per-leg draw from N(1, 0.06) clipped to [0.8, 1.25]. The agent observes only *expected* (pre-gust) energies; the realised use shows in its battery at the next decision. **Failure:** if a leg needs more than the remaining battery (or a parked drone drains to 0) the drone and its cargo are lost and the episode terminates (penalties under Reward). Weather is a 3-state Markov chain stepped every 10 min (P(storm → storm) = 0.70). Requests arrive per site as Poisson processes (hospital 1/60, pharmacy 1/100, home 1/90 per min). A waiting order expires when its deadline (request time + the type's window) passes; blood / vaccine spoil after 60 / 90 min on board. **Sites, orders, packages:** each site has at most one active order (one package); further requests meanwhile go to ground transport (counted, never observed). This modelling assumption, not a real-world limit, gives one load action and one feature block per site while keeping demand stochastic. A sortie carries up to 3 packages for different sites, one delivered per visit.", before=100)

    # --- Action space
    a = f.replace(9, "**Discrete, 25 actions**, each a real drone operation that consumes simulated time:")
    t = f.new_table(a, ["Action", "Meaning", "Real-world mapping", "Duration"], [
        ["0", "Hold", "Wait on pad / for weather", "5 min"],
        ["1–11", "Load order for site i", "Crew loads package into bay (depot only)", "1 min"],
        ["12–22", "Fly to site i", "Autonomous leg; package lowered on arrival", "distance/speed + 1 min"],
        ["23", "Return to depot", "Leg back to distribution centre", "distance/speed + 1 min"],
        ["24", "Charge", "Pad top-up, 2.5%/min; env stops it when full", "≤ 10 min (env-set)"],
    ], [900, 2000, 3729, 2400])
    f.para_after(t, "**Action masking.** Impossible actions (loading away from the depot or beyond capacity, flying to a site whose package is not on board, charging away from the pad or when nearly full) are masked. The 25-bit mask is supplied **separately** from the 108 features (Gymnasium Dict keys observation and action_mask): networks see only the features; the mask only decides what can be selected. It is applied in DQN acting and targets (masked argmax / max; ε-greedy and warm-up actions drawn from valid ones), in PG sampling and loss evaluation (MaskedCategorical under each stored mask) and in greedy evaluation. Risky-but-possible actions (flying on low battery) are not masked.",
                 before=100)

    # --- Observation space
    f.replace(11, "108 float32 values: 20 global features for the drone and mission (battery 1, time 1, weather 3, location 12, payload weight 1, payload count 1, energy to depot 1) plus 8 per destination, so 20 + 11 × 8 = 108. Future requests and the weather path are hidden. **All sources below are simulated:** they name the real-world interface each feature stands for, but every value comes from the simulator (no hardware, live APIs or medical systems).", after=100)
    obs = [
        ["battery", "State of charge", "Sensor: BMS", "float32 ÷ capacity", "[0, 1]"],
        ["time_remaining", "Share of shift left", "System: clock", "float32", "[0, 1]"],
        ["weather", "Calm / windy / storm", "API: weather", "one-hot (3)", "{0, 1}"],
        ["location", "Depot or site", "Sensor: GNSS", "one-hot (12)", "{0, 1}"],
        ["payload_weight", "Mass carried", "Sensor: load cell", "float32 ÷ 2.5 kg", "[0, 1]"],
        ["payload_count", "Packages on board", "System: manifest", "float32 ÷ 3", "[0, 1]"],
        ["energy_to_depot", "Energy to fly home", "Derived: model", "float32 ÷ capacity", "[0, 1]"],
        ["site.pending (×11)", "Order waiting", "API: orders", "binary", "{0, 1}"],
        ["site.on_board (×11)", "Package loaded", "System: manifest", "binary", "{0, 1}"],
        ["site.urgency (×11)", "Clinical urgency", "API: orders (triage)", "float32 ÷ 3", "[0, 1]"],
        ["site.value (×11)", "On-time value", "API: orders", "float32 ÷ 10", "[0, 1]"],
        ["site.deadline_slack (×11)", "Minutes to deadline", "API: orders", "float32 ÷ 120, clip", "[−1, 1]"],
        ["site.weight (×11)", "Package mass", "Dataset: manifest", "float32 ÷ 2.5 kg", "[0, 1]"],
        ["site.cold_chain_left (×11)", "Cold-chain life left", "Sensor: cold-box log", "float32 ÷ 90 min", "[0, 1]"],
        ["site.leg_energy (×11)", "Energy to fly there", "Derived: model", "float32 ÷ capacity", "[0, 1]"],
    ]
    f.fill_table(f.T[0]._tbl, ["Observation", "Description", "Source (Sensor/Camera/API/Dataset)", "Encoding / Data Type", "Range"],
                 obs, [2600, 1900, 1929, 1800, 800])
    after_obs = P[13]._p
    _plain_ppr(after_obs, 100, 100)
    _set_runs(after_obs, "**Package types** (urgency, value, weight, deadline from request, cold-chain limit): blood (3, 10, 0.5–1.2 kg, 40–60 min, 60 min); emergency kit (3, 9, 0.8–1.6 kg, 35–50 min, none); vaccine (2, 6, 0.6–1.5 kg, 90–120 min, 90 min); prescription (2, 4, 0.2–0.8 kg, 60–120 min, none); medicine (1, 3, 0.8–2.0 kg, 120–180 min, none). **Update cadence:** every feature is recomputed once per decision, when the previous action ends (1–20 simulated min apart). Battery changes after each leg, charge or parked wait; location on arrival (the viewer interpolates flights); weather every 10 min, seen at the next decision; deadline slack and cold-chain life from the clock at each decision; urgency, value and weight are fixed when the order arrives; energy estimates are recomputed each decision from node, payload and weather.")
    f.para_after(after_obs, "**Start state:** t = 0 at the depot, 100% battery, empty, 4–6 orders, weather from the stationary distribution. **Termination:** the shift ends (240 min) or the mission fails (battery flat in flight = crash, or parked = stranded). The clock is checked between decisions, so an action begun before minute 240 completes: the demo's last flight started at 235.6 and landed at 244.2. No requests arrive after 240, but deadlines still apply. **Truncation:** a 500-decision cap, never reached (episodes take about 40 decisions). At shift end, on-board packages are penalised, orders waiting at the depot are not, and the drone need not return but pays −5 if away. Demand and weather are pre-sampled per seed, so every policy faces identical scenarios.", before=0)

    # --- Reward
    r = f.replace(15, "Per decision step t:", after=40)
    r = f.paras(r, ["r_t = Σ_on-time v_k + 0.25 Σ_late v_k − 0.5 Σ_deadline-passed v_k − Σ_spoiled v_k − 0.02 E_t − 1[invalid]",
                    "        − 1[failure]·(50 + Σ_on-board v_k + 0.5 Σ_active v_k) − 0.5 Σ_on-board-at-shift-end v_k − 5·1[shift ends away from depot]"],
                after=0, italic=True)
    r = f.para_after(r, "v_k is the package value (10, 9, 6, 4, 3); E_t is the battery, in percentage points, used by flight in the step. **Each term fires once, on its event:** +v on an on-time delivery; −0.5 v when a deadline passes (a waiting order is cancelled, an on-board one becomes late); +0.25 v on a late delivery (net −0.25 v); −v on spoilage; on failure −50, −v per package lost and −0.5 v per waiting order not already penalised; the shift-end terms on the final transition; −1 per invalid action (mask off only). **Trade-offs:** reward comes only from deliveries and scales with clinical value (critical ≈ 3× routine); late keeps some value; energy (0.1–1.0 per leg) discourages wasted flights without outweighing deliveries; the failure penalty stops early crashing being used to avoid deadline penalties (seen in a prototype).", before=100)

    # --- Environment visualization (added sub-section, same heading style as Reward Structure)
    h = f.clone_label(14, r, " Environment Visualization")
    v = f.para_after(h, "**Integration.** play.py runs the selected checkpoint greedily (masked) in the Gymnasium environment. EpisodeRecorder (environment/rendering.py) writes one JSON record per decision through a sink; only the file sink is implemented (no live Server-Sent Events sink), so the viewer is **replay-only**. The Three.js [3] viewer fetches the file, rebuilds the state at time t from the previous snapshot plus events up to t, interpolates the drone between takeoff and arrive events, and fills its panels from reward, cumulative_reward and state. It does not run the agent, decide or simulate. A real record (scenario seed 42, step 2, trimmed): take-off in a storm, windy mid-leg, blood delivered on time. expected_energy = 38.06 is the pre-gust estimate; a gust factor ≈1.04 made the realised use 39.53 points (battery 100 → 60.47), which the energy term charges: +10 − 0.02 × 39.53 = 9.21. Delivery is logged on arrival (13.88); t_end is 14.88 because lowering the package (drop_min) takes 1 more minute:", before=0)
    v = f.para_after(v, STEP_JSON, before=0, after=80, size=14, font="Courier New")
    v.find(W("w:pPr")).find(W("w:jc")).set(W("w:val"), "left")   # no justified stretching inside the JSON
    v = f.para_after(v, "**Observed behaviour (scenario seed 42, not a test seed; 2M-frame DQN from training seed 2).** 35 requests: 14 went to ground transport because their site was busy, and 21 became orders: 11 on time, 0 late, 8 expired, 2 still waiting at shift end, 0 spoiled; 7 of the 14 critical requests were served on time; return 53.3. It flew 7 sorties, mostly a hospital plus a nearby home, recharging after almost each, and landed at minute 244 with 8% battery. Questionable habit: it sometimes loads and then charges before leaving (4 times).")
    f.figure(v, ROOT / "assets" / "viewer.png", 10.5,
             "Fig. 1 3D replay of the selected DQN agent (scenario seed 42): drone en route to Hospital North; reward, battery, payload and request panels.")

    # --- System analysis and design
    f.para_after(P[16]._p, "All agents use TorchRL 0.15 [2] on GymWrapper(MediFlightEnv), the same 108→128→128 tanh MLP, Adam, gradient-norm clipping and a 500k-frame budget; evaluation is the deterministic masked argmax.",
                 before=100)
    f.replace(18, "QValueActor maps the 108 features to 25 Q-values [4], masked before every argmax and max. **Exploration:** 5k warm-up frames of random valid actions, then ε falls linearly per collected frame from 1.0 to its final value (reference 0.05 over 100k frames) and stays there. **Replay:** uniform TensorDictReplayBuffer; 16 frames collected, then 4 gradient steps. **Loss / target:** smooth-L1 DQNLoss; hard target copy every 1,000 steps (soft Polyak in one run). The reference uses **Double DQN** [5]: the online network selects a′ and the target network evaluates it, reducing overestimation. The selected row 10 uses the vanilla max_a′ Q_target(s′, a′); it was 0.6 better on validation, within the seed spread, which is no evidence that vanilla DQN is generally better.")
    hp = P[19]._p
    hp.findall(".//" + W("w:t"))[0].text = "Policy Gradient Methods (REINFORCE, PPO, A2C)"
    pg = f.replace(20, "Shared: a ProbabilisticActor with a MaskedCategorical over 25 actions; a ValueOperator critic of the same architecture.", after=40)
    f.paras(pg, [
        "**REINFORCE [6]:** complete episodes, Monte-Carlo returns G_t, ReinforceLoss with Â_t = normalised G_t. Row 6 uses Â_t = G_t − V(s_t), with V trained **jointly** (same optimiser, squared error to G_t, weight 0.5); since V ignores the action it should lower gradient variance without biasing the gradient.",
        "**PPO [9]:** each update uses **2,048 frames in total** (4 envs × 512 steps), GAE λ = 0.95 [8], ClipPPOLoss (ε = 0.2, value 0.5, entropy), 4 epochs × 8 minibatches of 256.",
        "**A2C [7]:** 8 envs × 8 steps per update, one gradient step (A2CLoss). GAE λ = 1 gives the discounted rewards to the end of the 8-step rollout plus the critic's value there (no bootstrap past an episode end), minus V(s_t); λ = 0 (row 5) is the one-step TD error r_t + γV(s_t+1) − V(s_t), relying entirely on the critic.",
    ], after=40)

    # --- Implementation
    f.para_after(P[21]._p, "**Protocol.** Per algorithm, **10 masked configurations** (reference row 1 plus 9 variations, rows 2–10: the required ten) and an **extra** no-mask ablation (row 11). Each variation changes one hyperparameter to isolate its effect, which cannot reveal interactions. Every run: 500k frames, 3 seeds (0–2). **Checkpoint selection, identical for all algorithms:** every 20k frames the greedy return on the first 20 validation seeds; the best checkpoint is kept. **Mean Reward** = that checkpoint on all 50 validation seeds (10000–10049), **mean ± SD across the 3 training seeds**; the highest is bold. Test seeds (20000–20199) were never used here. γ = 0.99 unless shown. On these seeds EDF scores 23.8 and nearest-first 8.8.", before=100)

    E1 = "ε 1→0.05 / 100k"
    dqn_rows = [
        ["5e-4", "0.99", "100,000", "128", E1, "hard /1000, Double", "35.11 ± 2.47"],
        ["1e-4", "0.99", "100,000", "128", E1, "hard /1000, Double", "27.16 ± 1.92"],
        ["2e-3", "0.99", "100,000", "128", E1, "hard /1000, Double", "32.19 ± 2.99"],
        ["5e-4", "0.95", "100,000", "128", E1, "hard /1000, Double", "34.23 ± 1.82"],
        ["5e-4", "0.99", "10,000", "128", E1, "hard /1000, Double", "34.17 ± 3.14"],
        ["5e-4", "0.99", "100,000", "32", E1, "hard /1000, Double", "31.03 ± 1.88"],
        ["5e-4", "0.99", "100,000", "128", "ε 1→0.01 / 25k", "hard /1000, Double", "32.74 ± 2.24"],
        ["5e-4", "0.99", "100,000", "128", "ε 1→0.1 / 250k", "hard /1000, Double", "36.12 ± 0.88"],
        ["5e-4", "0.99", "100,000", "128", E1, "soft τ 0.005, Double", "33.60 ± 1.56"],
        ["5e-4", "0.99", "100,000", "128", E1, "hard /1000, vanilla", "36.76 ± 0.49"],
        ["5e-4", "0.99", "100,000", "128", E1, "Double, no mask", "19.71 ± 4.24"],
    ]
    f.fill_table(f.T[1]._tbl, ["Learning Rate", "Gamma", "Replay Buffer", "Batch Size", "Exploration (ε, frames)", "Target Update", "Mean Reward ± SD"],
                 dqn_rows, [1000, 750, 1050, 750, 1750, 2079, 1650], bold_rows={9})
    f.para_after(f.T[1]._tbl, "**Observations.** Learning rate mattered most (1e-4: 27.2, too slow; 2e-3: 32.2 ± 3.0, noisy); batch 32 lost about 4. Rows 8 (36.12 ± 0.88) and 10 (36.76 ± 0.49) were best, but the 0.6 gap is below the seed SD, so row 10's advantage is **not statistically established**. Without the mask: 19.7, with about 2.9 invalid actions per episode.", before=100)

    rf_rows = [
        ["1e-3", "0.99", "8", "none", "yes", "0", "22.01 ± 3.82"],
        ["3e-4", "0.99", "8", "none", "yes", "0", "18.58 ± 0.31"],
        ["3e-3", "0.99", "8", "none", "yes", "0", "22.35 ± 3.54"],
        ["1e-3", "0.95", "8", "none", "yes", "0", "22.53 ± 4.48"],
        ["1e-3", "0.999", "8", "none", "yes", "0", "22.10 ± 3.65"],
        ["1e-3", "0.99", "8", "learned V(s)", "yes", "0", "18.25 ± 1.12"],
        ["1e-3", "0.99", "8", "none", "no", "0", "18.97 ± 1.57"],
        ["1e-3", "0.99", "2", "none", "yes", "0", "22.77 ± 4.04"],
        ["1e-3", "0.99", "32", "none", "yes", "0", "19.02 ± 0.80"],
        ["1e-3", "0.99", "8", "none", "yes", "0.01", "19.12 ± 3.14"],
        ["1e-3", "0.99", "8, no mask", "none", "yes", "0", "−46.32 ± 0.00"],
    ]
    f.fill_table(f.T[2]._tbl, ["Learning Rate", "Gamma", "Episodes / Update", "Baseline", "Return Norm.", "Entropy Coef.", "Mean Reward ± SD"],
                 rf_rows, [1100, 900, 1350, 1400, 1150, 1379, 1750], bold_rows={7})
    f.para_after(f.T[2]._tbl, "**Observations.** A narrow band (18–23) with large seed spread; the selected row 8 is not meaningfully better than the reference. Under this budget and implementation the learned baseline, larger batches and entropy bonus gave steadier but lower returns, which does not show that baselines hurt in general. **Without the mask every seed collapsed to always Hold** (−46.3, deadline penalties only). Without the mask invalid actions can still be chosen but cost −1 and a wasted minute; Hold is the only always-valid action, and early invalid actions and crashes (−50) probably pushed probability onto it before any delivery was found.", before=100)

    ppo_rows = [
        ["3e-4", "0.2", "4", "0.95", "0.01", "2048 / 256", "24.35 ± 3.07"],
        ["1e-4", "0.2", "4", "0.95", "0.01", "2048 / 256", "18.98 ± 1.97"],
        ["1e-3", "0.2", "4", "0.95", "0.01", "2048 / 256", "28.04 ± 1.29"],
        ["3e-4", "0.1", "4", "0.95", "0.01", "2048 / 256", "28.12 ± 1.49"],
        ["3e-4", "0.3", "4", "0.95", "0.01", "2048 / 256", "24.57 ± 1.23"],
        ["3e-4", "0.2", "10", "0.95", "0.01", "2048 / 256", "30.95 ± 1.23"],
        ["3e-4", "0.2", "4", "0.80", "0.01", "2048 / 256", "26.31 ± 2.70"],
        ["3e-4", "0.2", "4", "0.95", "0", "2048 / 256", "23.58 ± 0.69"],
        ["3e-4", "0.2", "4", "0.95", "0.05", "2048 / 256", "28.58 ± 0.64"],
        ["3e-4", "0.2", "4", "0.95", "0.01", "512 / 128", "24.56 ± 2.90"],
        ["3e-4", "0.2", "4", "0.95", "0.01", "2048 / 256, no mask", "−15.03 ± 21.68"],
    ]
    ppo_tbl = f.T[3]._tbl
    a2c_proto = copy.deepcopy(ppo_tbl)
    f.fill_table(ppo_tbl, ["Learning Rate", "Clip ε", "Epochs", "GAE λ", "Entropy Coef.", "Rollout (total) / Minibatch", "Mean Reward ± SD"],
                 ppo_rows, [1100, 850, 850, 850, 1100, 2379, 1900], bold_rows={5})
    po = f.para_after(ppo_tbl, "**Observations.** The most sensitive algorithm (19.0–31.0). More use of each rollout helped: 10 epochs +6.6 (80 instead of 32 minibatch updates per rollout), learning rate 1e-3 +3.7, clip 0.1 +3.8, entropy 0.05 +4.2. Heavy reuse can over-fit a rollout; the clip limits this and it did not dominate here, though it was not measured directly. No mask: seed-dependent (−15.0 ± 21.7).",
                      before=100)

    a2c_label = f.clone_label(28, po, "A2C")
    po.addnext(a2c_label)  # (clone_label already inserted after po)
    a2c_tbl = a2c_proto
    a2c_label.addnext(a2c_tbl)
    a2c_rows = [
        ["7e-4", "8 × 8", "1.0", "0.01", "0.5", "no", "25.21 ± 0.52"],
        ["2e-4", "8 × 8", "1.0", "0.01", "0.5", "no", "25.03 ± 0.73"],
        ["2e-3", "8 × 8", "1.0", "0.01", "0.5", "no", "23.55 ± 1.54"],
        ["7e-4", "8 × 8", "0.95", "0.01", "0.5", "no", "24.53 ± 1.76"],
        ["7e-4", "8 × 8", "0 (1-step TD)", "0.01", "0.5", "no", "16.00 ± 2.14"],
        ["7e-4", "8 × 32", "1.0", "0.01", "0.5", "no", "21.11 ± 3.78"],
        ["7e-4", "8 × 8", "1.0", "0", "0.5", "no", "24.82 ± 1.18"],
        ["7e-4", "8 × 8", "1.0", "0.05", "0.5", "no", "23.32 ± 1.19"],
        ["7e-4", "8 × 8", "1.0", "0.01", "0.25", "no", "24.44 ± 1.12"],
        ["7e-4", "8 × 8", "1.0", "0.01", "0.5", "yes", "24.40 ± 0.70"],
        ["7e-4", "8 × 8, no mask", "1.0", "0.01", "0.5", "no", "−46.32 ± 0.00"],
    ]
    f.fill_table(a2c_tbl, ["Learning Rate", "Rollout (Envs × Steps)", "GAE λ", "Entropy Coef.", "Critic Coef.", "Adv. Norm.", "Mean Reward ± SD"],
                 a2c_rows, [1100, 1700, 1450, 1050, 1000, 979, 1750], bold_rows={0})
    f.para_after(a2c_tbl, "**Observations.** The least sensitive *among the 9 variations tested* (7 within 2 points); the reference was selected. λ = 0 lost 9 points: it relies on a still-inaccurate critic, while λ = 1 uses up to 8 real rewards before bootstrapping. 8 × 32 rollouts cost 4 points: with one update per rollout, 500k frames allow about 1,950 updates instead of 7,800. No mask: always Hold, as REINFORCE.",
                 before=100)

    # --- Results discussion: protocol + CI method
    f.replace(32, "**How the results were produced**", before=100, after=40)
    f.paras(P[32]._p, [
        "Seeds are disjoint; selection used only validation seeds. Each selected configuration got 2 more seeds (3–4) and the 5 models were tested **once** on 200 test seeds (20000–20199); scenarios are pre-sampled per seed, so comparisons are paired. **Uncertainty:** ± SD across training seeds (sweeps, seed spread); ± 95% CI over the 200 test scenarios (each seed's return averaged over the 5 models, then mean ± 1.96·s/√200); paired bootstrap 95% CI of the difference from EDF (10,000 resamples). Wins = share of seeds where the agent scored higher. **Baselines** use the agents' actions, constraints and current-state information. *EDF (earliest deadline first)* ranks waiting orders by on-time feasibility, urgency and deadline, loads while the sortie stays on time with a 15% energy margin and 5% reserve, flies in deadline order and charges when the next sortie is infeasible. *Nearest-first* serves the nearest orders while battery-feasible. *Random* picks uniformly among valid actions.",
    ], after=100)
    empty33 = P[33]._p
    empty33.getparent().remove(empty33)

    # --- Cumulative rewards (P34 label kept; P35 held both the placeholder and the "Training Stability" label)
    c = f.replace(35, "**Training (Fig. 2).** All methods start at −50 to −65 (orders expiring and frequent battery failures; masked random: −65.8, 97% crashes) and rise above nearest-first. Training returns come from exploratory behaviour (DQN's ε ≥ 0.05; sampling for PG methods), validation from the greedy masked argmax: over the last 100k frames (seed 0), training vs validation was 27 vs 35 (DQN), 26 vs 32 (PPO), 19 vs 23 (A2C) and 20 vs 20 for the already near-deterministic REINFORCE.", after=40)
    c = f.figure(c, FIG / "fig1_cumulative_rewards.png", 11.0,
                 "Fig. 2 Cumulative reward per episode over training episodes, best configuration per method (thin: training, per seed; thick: greedy validation; 5 seeds).")
    c = f.para_after(c, "**Validation** (optimistic, used for selection): checkpoints picked on the first 20 validation seeds, then evaluated once on all 50 (± SD over 5 models): DQN 35.75 ± 2.05, PPO 30.83 ± 1.07, A2C 25.28 ± 0.93, REINFORCE 23.08 ± 2.91; EDF 23.78, nearest-first 8.79. **Test** (held-out, 200 seeds). *Critical on time* = critical requests delivered on time ÷ all critical requests in the shift, averaged over scenarios:")
    c = f.new_table(c, ["Policy", "Return ± 95% CI", "Paired diff. vs EDF [95% CI]", "Wins vs EDF", "On-time", "Late", "Critical on time", "Crash rate"], [
        ["DQN", "33.88 ± 1.72", "+12.32 [+10.42, +14.26]", "84.5%", "9.77", "1.04", "33.2%", "1.9%"],
        ["PPO", "26.42 ± 1.69", "+4.86 [+2.98, +6.81]", "66.5%", "9.03", "0.68", "30.6%", "2.4%"],
        ["A2C", "21.52 ± 2.05", "−0.03 [−2.34, +2.39]", "51.5%", "8.71", "0.86", "26.9%", "1.9%"],
        ["REINFORCE", "20.21 ± 1.57", "−1.35 [−3.17, +0.58]", "39.0%", "8.49", "0.73", "26.4%", "0.7%"],
        ["EDF", "21.56 ± 2.01", "—", "—", "9.88", "0.01", "23.4%", "1.5%"],
        ["Nearest-first", "3.32 ± 2.47", "—", "—", "10.67", "0.30", "8.1%", "5.5%"],
        ["Random", "−65.75 ± 2.00", "—", "—", "3.20", "0.46", "11.5%", "97.0%"],
    ], [1450, 1350, 2200, 750, 750, 600, 1050, 879], bold_rows={0})
    c = f.paras(c, [
        "*Training-seed spread of the test return (SD of 5 models):* DQN 0.75, PPO 1.43, A2C 0.76, REINFORCE 2.23. **Findings.** (1) **DQN performed best:** +12.3 over EDF [+10.4, +14.3], winning 84.5% of scenarios; PPO also beat EDF (+4.9 [+3.0, +6.8]). (2) **A2C and REINFORCE are statistically indistinguishable from EDF.** (3) **Triage, not volume:** DQN delivered slightly fewer orders on time than EDF (9.77 vs 9.88) but served critical requests on time 33.2% vs 23.4% (paired +9.8 points [+8.6, +11.1]), at the cost of (4) about one late delivery per shift (1.04 vs 0.01).",
    ], after=60)

    # --- Training stability (new label, cloned from "Cumulative Rewards")
    ts_label = f.clone_label(34, c, "Training Stability")
    _set_space = lambda el, b: el.find(W("w:pPr")).insert(0, el.makeelement(W("w:spacing"), {W("w:before"): str(b), W("w:after"): "40", W("w:lineRule"): "auto"}))
    _set_space(ts_label, 160)
    s = f.replace(36, "**DQN (Fig. 3).** The TD loss falls from 4.8 to 1.7 by 40k frames, then to about 1.3. Its targets move with the network, so this alone does not prove a better policy; the rising validation return and test results do. The mean Q of the taken action peaks near 20 at about 200k frames and settles at 14–15: a change in *estimated* values, consistent with returns growing after ε-annealing ends and hard target updates correcting them. No seed diverged (test SD 0.75).", after=40)
    ts_label.getparent().remove(ts_label); P[36]._p.addprevious(ts_label)
    s = f.figure(s, FIG / "fig3_dqn_objective.png", 10.5, "Fig. 3 DQN TD loss and mean Q of the taken action (5 seeds).")
    s = f.para_after(s, "**Policy entropy (Fig. 4)**: high means probability spread over many valid actions (exploration), low means near-deterministic. REINFORCE fell from 1.5 to 0.2 nats by 50k frames (0.04 at the end), committing early (widest seed spread, SD 2.23). A2C fell to about 0.1, stabilised by its critic (SD 0.76). PPO stayed highest (about 0.28) because clipping limits each update, and was still improving at the end (SD 1.43).")
    f.figure(s, FIG / "fig4_pg_entropy.png", 8.0, "Fig. 4 Policy entropy of the masked policies (mean, min–max over 5 seeds).")

    # --- Episodes to converge
    e = f.replace(38, "One definition for all algorithms (table and Fig. 5): the first evaluation at which the 3-point moving average of the validation return reaches 90% of the way from its initial value to its final plateau (mean of the last 5 evaluations). Statistics are over the 5 training seeds (0–4) of each selected configuration; frames are means rounded to the nearest 1k. Episodes and frames differ in ratio because episode length depends on the policy's decisions and on crashes (about 30–36 decisions per episode here).", before=40, after=100)
    for rr in P[38]._p.findall(W("w:r")):
        rp_ = rr.find(W("w:rPr")); sz = rp_.find(W("w:sz")); sz.set(W("w:val"), "20"); rp_.find(W("w:szCs")).set(W("w:val"), "20")
    e = f.new_table(e, ["", "A2C", "REINFORCE", "PPO", "DQN"], [
        ["Episodes (mean ± SD, 5 seeds)", "2,275 ± 630", "3,364 ± 2,243", "4,299 ± 844", "6,055 ± 1,203"],
        ["Frames (mean, ≈)", "80k", "120k", "153k", "180k"],
    ], [2829, 1550, 1550, 1550, 1550])
    p39 = P[39]._p
    _plain_ppr(p39, 100, 40)
    _set_runs(p39, "**A2C converged fastest but to the lowest plateau**; REINFORCE varied most (1,662–7,218 episodes); **DQN was slowest but reached the highest plateau**, exploring for its first 100k frames. This is convergence to each algorithm's *own* plateau, so fast is not good.")
    e.addnext(p39)
    f.figure(p39, FIG / "fig5_convergence.png", 8.0, "Fig. 5 Episodes to reach 90% of the final validation plateau (dots: seeds).")
    P[40]._p.getparent().remove(P[40]._p)

    # --- Generalization
    g = f.replace(42, "Every test seed is an unseen initial state. As a further stress test, the models ran unchanged under four settings never seen in training, each changing one thing: arrival rates ×1.5 or ×0.6; +0.15 probability of moving to storm from every weather state (rows renormalised); 8–10 initial orders instead of 4–6. These reuse the **first 100 of the 200 test seeds**, so they are not an independent test set. Mean return over 5 models; ± is each model's 95% CI over the 100 scenarios (1.96·s/√100), averaged over the 5 models (computed per model, unlike the main test table):", before=40, after=100)
    for rr in P[42]._p.findall(W("w:r")):
        rp_ = rr.find(W("w:rPr")); rp_.find(W("w:sz")).set(W("w:val"), "20"); rp_.find(W("w:szCs")).set(W("w:val"), "20")
    g = f.new_table(g, ["Scenario", "DQN", "PPO", "A2C", "REINFORCE", "EDF", "Nearest-first"], [
        ["Nominal", "34.7 ± 3.5", "26.8 ± 3.3", "21.5 ± 3.3", "20.0 ± 3.0", "20.6 ± 2.8", "4.1 ± 3.2"],
        ["Arrivals ×1.5", "27.9 ± 4.2", "22.9 ± 3.8", "16.2 ± 3.6", "15.2 ± 3.4", "15.6 ± 2.6", "−0.3 ± 2.8"],
        ["Arrivals ×0.6", "40.0 ± 3.2", "32.2 ± 3.0", "30.2 ± 2.9", "29.3 ± 2.5", "31.8 ± 2.5", "14.6 ± 3.3"],
        ["Storm +0.15", "27.0 ± 3.5", "20.3 ± 3.2", "13.2 ± 3.8", "13.9 ± 3.0", "16.5 ± 2.5", "−6.7 ± 4.7"],
        ["8–10 initial orders", "32.9 ± 3.6", "26.1 ± 3.0", "21.6 ± 2.9", "19.2 ± 2.8", "20.3 ± 2.5", "−0.2 ± 3.9"],
    ], [1829, 1130, 1130, 1130, 1500, 1130, 1180])
    g = f.figure(g, FIG / "fig6_generalisation.png", 9.0, "Fig. 6 Return minus EDF's return under shifted conditions (first 100 test seeds, mean over 5 models).")
    g = f.para_after(g, "**DQN's mean return exceeded EDF's in every condition** (by 8.2–14.1; the reported intervals do not overlap, but no paired test was run here; crashes 0.8–2.6%). **PPO's lead was not robust:** it vanished under low demand and overlaps under storms. **A2C and REINFORCE fell below EDF in storms**; nearest-first crashed in 13% of stormy shifts. **Why it is harder:** storms raise leg energy 1.8× and cut speed to 0.7×, so sorties cost more battery and charging time and fewer deadlines are met; high demand exceeds what 3 slots and 2.5 kg can carry, so more orders expire.", before=100)

    # --- Follow-up experiment (label cloned from "Generalization")
    fu = f.clone_label(41, g, "Follow-up Experiment: a Larger Training Budget")
    fu = f.para_after(fu, "A **targeted follow-up, not an equal-budget comparison**: only DQN and PPO, whose validation curves were still rising at 500k frames (A2C and REINFORCE had converged by about 80k–120k), were trained to 2M frames as 5 new runs (seeds 0–4, trained from scratch, not continued from the 500k models; DQN also with a 2×256 network). Selection on validation only: DQN 35.75 → 41.03 → **42.98** (2×256); PPO 30.83 → **34.63**. The winners were then tested on the **same test scenarios already used once**, so this is additional evidence, not an independent final test; the 500k results remain the primary comparison.", before=40)
    fu = f.new_table(fu, ["Policy (2M frames)", "Test return ± 95% CI", "vs EDF [paired 95% CI]", "vs own 500k [paired 95% CI]", "Critical on time", "Late / shift"], [
        ["DQN, 2×256", "39.79 ± 1.72", "+18.23 [+16.36, +20.13]", "+5.92 [+4.62, +7.19]", "36.7%", "0.52"],
        ["PPO", "32.12 ± 1.73", "+10.56 [+8.62, +12.56]", "+5.71 [+4.44, +6.98]", "33.4%", "0.74"],
    ], [1350, 1400, 2250, 2029, 1000, 1000], bold_rows={0})
    f.para_after(fu, "Both gained about 6 points; DQN halved its late deliveries and beat EDF in 93% of scenarios. Best ch"
        "eckpoints came at 1.2–1.98M frames: the budget was limiting here, not proof that more training alway"
        "s helps; 2×256 vs 2×128 (43.0 vs 41.0) is within seed noise. The demo model is training seed 2 (best"
        " on validation), shown on scenario seed 42.", before=100)

    # --- Conclusion
    k = f.replace(44, "**Equal budget (500k frames, primary result).** **DQN** was strongest on held-out scenarios (33.9 ± "
        "1.7 vs 21.6 for EDF; +12.3 [+10.4, +14.3] paired), the most consistent across seeds and ahead of EDF"
        " in every shifted condition; PPO was second (+4.9); A2C and REINFORCE only matched EDF. **Extended b"
        "udget (separate follow-up, test set reused):** both gained about 6 points at 2M frames.", before=100, after=100)
    for rr in P[44]._p.findall(W("w:r")):
        rp_ = rr.find(W("w:rPr")); rp_.find(W("w:sz")).set(W("w:val"), "20"); rp_.find(W("w:szCs")).set(W("w:val"), "20")
    k = f.paras(k, [
        "**Why DQN did best** (plausible, not isolated experimentally): short episodes, discrete actions and "
        "delivery rewards suit off-policy replay (123,752 gradient steps in 500k frames), and PPO's best vari"
        "ations also increased reuse; REINFORCE became near-deterministic within 50k frames; A2C's critic mad"
        "e it stable but conservative.",
        "**Strengths.** Learned **clinical triage** (critical on time: EDF 23%, DQN 33% at 500k, 37% at 2M); "
        "masking was essential.",
        "**Weaknesses.** (1) *Lateness:* about 1 late delivery per shift for DQN at 500k (0.5 at 2M; EDF 0.01). (2) *End-of-shift reward weakness:* only on-board cargo is penalised at shift end (−0.5 v); waiting orders cost nothing. On seed 42 the 500k DQN loaded blood (deadline minute 269) at 233.7 and then charged, taking −5 where a 9-minute flight would have earned about +10: not a profitable exploit, but a mistake the reward makes cheap. (3) Agents optimise the reward they are scored on; EDF optimises deadlines. (4) Overloaded by design: about 26% of requests are served on time.",
        "**Limitations.** 500k-frame CPU training; 3 seeds per sweep; one-factor tuning; simplified physics; "
        "simulated, uncalibrated demand and weather; one drone; no real flights; stress tests and follow-up r"
        "euse test scenarios. **Future work (not implemented):** penalise orders pending at shift end; strict"
        "er critical deadlines; prioritised replay, dueling heads; multi-drone dispatch; realistic flight con"
        "straints; testing on newly generated scenarios.",
    ], after=100)

    # --- References (unnumbered heading cloned from "Conclusion and Discussion")
    ref_h = copy.deepcopy(P[43]._p)
    for child in list(ref_h):
        if child.tag != W("w:pPr"):
            ref_h.remove(child)
    rppr = ref_h.find(W("w:pPr"))
    for e_ in rppr.findall(W("w:numPr")):
        rppr.remove(e_)
    num = rppr.makeelement(W("w:numPr"), {})
    num.append(num.makeelement(W("w:ilvl"), {W("w:val"): "0"})); num.append(num.makeelement(W("w:numId"), {W("w:val"): "0"}))
    st_ = rppr.find(W("w:pStyle"))
    (st_.addnext(num) if st_ is not None else rppr.insert(0, num))
    src_r = P[43]._p.find(W("w:r"))
    rr_ = ref_h.makeelement(W("w:r"), {})
    if src_r is not None and src_r.find(W("w:rPr")) is not None:
        rr_.append(copy.deepcopy(src_r.find(W("w:rPr"))))
    te_ = rr_.makeelement(W("w:t"), {}); te_.text = "References"; rr_.append(te_); ref_h.append(rr_)
    k.addnext(ref_h)
    f.paras(ref_h, REFERENCES, after=20, size=16)

    # pagination: keep every heading with the paragraph that follows it (the template switches this off,
    # which leaves orphaned headings at page bottoms); let body paragraphs split across pages
    body = f.doc.element.body
    for p in body.iter(W("w:p")):
        ppr = p.find(W("w:pPr"))
        if ppr is None:
            continue
        st = ppr.find(W("w:pStyle"))
        is_heading = st is not None and st.get(W("w:val"), "").startswith("Heading")
        for kl in ppr.findall(W("w:keepLines")):
            if not is_heading:
                ppr.remove(kl)
        if is_heading:
            for kn in ppr.findall(W("w:keepNext")):
                kn.set(W("w:val"), "1")
    for idx in (8, 22, 28, 34, 37, 41):                 # bold/plain sub-labels that act as headings
        ppr = P[idx]._p.find(W("w:pPr"))
        if ppr is not None and ppr.find(W("w:keepNext")) is None:
            ppr.insert(0, ppr.makeelement(W("w:keepNext"), {W("w:val"): "1"}))

    # empty placeholder paragraphs of the template (they only add blank lines / an empty numbered heading)
    for idx in (12, 23, 25, 26, 27, 29, 30, 45, 46, 47):
        el = P[idx]._p
        if el.getparent() is not None and not "".join(el.itertext()).strip():
            el.getparent().remove(el)

    f.save(out)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()
    fill(a.template, a.out, a.video)


if __name__ == "__main__":
    main()
