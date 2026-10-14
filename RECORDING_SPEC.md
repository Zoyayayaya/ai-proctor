# Recording spec — what to record, and why

**Read this before you start. Then follow the prompts the recorder prints.**

| | |
|---|---|
| Total effort | **~50 minutes** for two people |
| Clips | 48 (8 scenarios × 3 repeats × 2 people) |
| Footage | ~10 minutes |
| Disk | well under 300 MB |
| Result | a labelled dataset the pipeline turns into the paper's numbers |

The recorder walks you through every scenario and names the files correctly, so
the only things you have to get right are **framing** and **holding the
behaviour long enough**. Those two are covered below.

---

## 1. Why these eight scenarios

You do not need this section to record — skip it if you like. It is here so the
scenarios do not look arbitrary.

The system measures seven things:

| Feature | What it actually responds to |
|---|---|
| `eye_ratio` | head or gaze turned **left / right** |
| `head_tilt` | head **rolled** sideways, ear toward shoulder |
| `body_tilt` | **leaning** left or right |
| `looking_up` | nose **high** in the frame |
| `hand_on_table` | hands **below the desk line** |
| `phone_detected` | a **phone visible** to the camera |
| `duration` | anything **sustained** rather than momentary |

The score needs **two or more** of these at once, or a visible phone, or a
behaviour held long enough. So each cheating scenario below is built to drive at
least two of them — otherwise we would be testing the system on something it has
no sensor for.

> **One honest caveat.** The system has **no head-pitch measurement**.
> `head_tilt` is *roll* — ear toward shoulder — not how far the head is nodded
> down. So **S6, "looking down at notes", is genuinely hard for this system and
> is expected to score poorly.** Record it anyway. It is the most common real
> cheating behaviour, and reporting it honestly as the weakest case is worth far
> more than quietly testing only the cases that work. **Do not re-record S6
> because it looks wrong — a low score there is a finding, not a fault.**

---

## 2. Camera setup — do this once per person

| | |
|---|---|
| Camera | Laptop webcam or phone. **Landscape.** 1280×720 or better. |
| Distance | ~60–80 cm back, so your **head and shoulders fill roughly the middle half** of the frame. |
| Lighting | Even, from in front. **No window behind you.** |
| Framing | Whole head visible with some space above it. Shoulders in shot. |
| Desk | A desk edge visible helps — the hand rule keys off the lower third of the frame. |
| Background | Plain wall is best. One person only. |
| Angle | Straight on, roughly at eye level. **Do not put the camera off to one side** — that alone shifts `eye_ratio` off centre and creates false positives. |

**Do not move the seat, camera or lighting part-way through a person's session.**
Changing them mid-session creates signal that is not behaviour.

---

## 3. The eight scenarios

**Three repeats of each. Hold the behaviour for at least 6 seconds, inside a
12-second clip.**

The hold length matters: the duration bonus builds at 4 points per second, so a
one-second glance is *correctly ignored by design* and would end up labelled
wrongly.

### Safe — the system must NOT flag these

| Code | Name | Do exactly this |
|---|---|---|
| **S1** | Focused | Face the screen, eyes on it, sit naturally still. Blink normally. |
| **S2** | Fidgeting | Shift your weight, adjust posture, scratch your face, stretch, push your hair back. **Do not** turn your head away, tilt it, or look down for long. |
| **S3** | Reading | Eyes on the screen, small nods, lean in slightly to read. |

### Cheating — the system SHOULD flag these

| Code | Name | Do exactly this |
|---|---|---|
| **S4** | Look away | Turn your head clearly **left or right**, past your shoulder line, and **hold ~6 s**. Alternate sides across the three repeats. |
| **S5** | Head roll | **Roll your head sideways** like stretching your neck — ear toward shoulder — and hold ~6 s. Alternate sides. |
| **S6** | Look down at notes | Tilt your head down as if reading notes in your lap. Hold ~6 s. **The hard one — see the caveat above.** |
| **S7** | Phone visible | Bring a phone **up into the camera's view** — above the desk, clearly visible — and hold ~6 s. Not hidden behind the desk. |
| **S8** | Lean / leave frame | Lean out sideways so your face partly leaves the frame, or stand and step out, for ~6 s, then return. |

> **S7 matters most.** A visible phone is the single feature that made the
> original paper's accuracy figure unreliable, so clean, clearly-visible phone
> events are what let us measure it properly.

---

## 4. What the recorder does

```powershell
uv run python scripts/live_test.py --record --person 1
```

For each of the 24 clips it prints the scenario, waits for you to press Enter,
counts down 3-2-1, records 12 seconds, and saves the file with the correct name.

**The filename is the label.** `S7_p1_r2.mp4` means "phone visible, person 1,
repeat 2". Nothing else records what happened, so a misnamed file is worse than
a missing one — which is why the recorder names them for you.

Ctrl-C aborts safely at any point.

Then repeat for each person:

```powershell
uv run python scripts/live_test.py --record --person 2
```

**Two people minimum.** Three or four is better and strengthens the
cross-validation, which is the specific thing the reviewer objected to.

---

## 5. Check before you process

```powershell
uv run python scripts/check_clips.py --people 1,2
```

It reports, per clip: resolution, duration, **the fraction of frames containing a
detectable face**, which indicators fire, and a scenario × person × repeat grid.
Then PASS or FAIL.

**Do not skip this.** If it says FAIL, fix those clips first. The usual cause is
the face-detection fraction — you were too far away, too dark, or off centre.
Re-record just those. Processing unusable clips wastes an hour and produces
numbers that mean nothing.

Expected output: `PASS - the clips are ready to process`.

---

## 6. Then run the pipeline

```powershell
uv run python scripts/extract_features.py      # clips  -> data/dataset_labelled.csv
uv run python scripts/eval.py                  #        -> data/results.json
uv run python scripts/make_result_figures.py   #        -> figures/ at 600 dpi
```

That is the whole loop. The metrics are also sent back automatically when
`eval.py` finishes, so nobody has to copy files around — see `RUNBOOK.md` §2.7.

If anything goes wrong, `RUNBOOK.md` §9 lists the common symptoms and fixes.
