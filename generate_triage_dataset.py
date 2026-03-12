"""
MediTriage-LLM: Synthetic Triage Dataset Generator v2
=====================================================
Generates realistic patient portal messages with structured triage labels.

v2 changes:
  - Heavily improved Stage 2 prompt with concrete behavioral guidance per dimension
  - Fixed on_behalf_of labels (for_self / for_my_child / for_my_parent / for_my_spouse)
  - Removed detail_level (now derived from health_literacy × message_type naturally)
  - Configurable model (free or paid)
  - ThreadPoolExecutor concurrency for Stage 2

Usage:
    pip install requests
    export OPENROUTER_API_KEY="your-key-here"
    python generate_triage_dataset.py --stage 1
    python generate_triage_dataset.py --stage 2
    python generate_triage_dataset.py --stage 3
"""

import os
import json
import time
import random
import argparse
import requests
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# --- Model Selection ---
# Free:  "stepfun/step-3.5-flash:free"        → $0 total
# Cheap: "google/gemini-2.0-flash-001"         → ~$0.30 total
# Cheap: "openai/gpt-4.1-nano"                 → ~$0.30 total
# Good:  "openai/gpt-4.1-mini"                 → ~$1.20 total
MODEL = "openai/gpt-4.1-nano"

MAX_RETRIES = 3
RETRY_DELAY = 5

# --- Concurrency ---
MAX_WORKERS = 4          # Free-tier: 3-4. Paid: 8-15.
STAGGER_DELAY = 0.3      # seconds between thread launches in a batch

# --- Output Paths ---
OUTPUT_DIR = Path("data/raw")
CLUSTERS_FILE = OUTPUT_DIR / "symptom_clusters.json"
RAW_DATASET_FILE = OUTPUT_DIR / "raw_dataset.json"
FINAL_DATASET_DIR = OUTPUT_DIR / "final"

# --- Generation Parameters ---
CLUSTERS_PER_COMBO = 5
MESSAGES_PER_CLUSTER = 7
RANDOM_SEED = 42

# ============================================================================
# DEPARTMENTS & URGENCY
# ============================================================================

DEPARTMENTS = [
    "Emergency Medicine",
    "General Internal Medicine / Primary Care",
    "Cardiology",
    "Orthopedics",
    "Neurology",
    "Gastroenterology",
    "Pulmonology",
    "Dermatology",
    "Psychiatry / Behavioral Health",
    "Obstetrics & Gynecology",
    "Urology",
    "ENT / Otolaryngology",
    "Endocrinology",
    "Ophthalmology",
    "Nephrology",
    "Infectious Disease",
    "Allergy & Immunology",
    "Rheumatology",
    "Dentistry / Oral Surgery",
    "Oncology",
]

URGENCY_MAP = {"Emergency Medicine": ["Emergency"]}
DEFAULT_URGENCY_LEVELS = ["Low", "Medium", "High"]

# --- Patient Profile Dimensions (v2: clearer labels) ---
HEALTH_LITERACY_OPTIONS = ["low", "medium", "high"]
EMOTIONAL_TONE_OPTIONS = ["calm", "anxious", "frustrated", "panicked"]
ON_BEHALF_OF_OPTIONS = ["for_self", "for_my_child", "for_my_parent", "for_my_spouse"]
MESSAGE_TYPE_OPTIONS = ["symptom_report", "health_question", "follow_up", "medication_concern"]
WRITING_STYLE_OPTIONS = ["formal", "casual", "texting-style"]
# NOTE: detail_level removed in v2 — length is now driven by literacy × message_type

TONE_TO_SENTIMENT = {
    "calm": "Calm",
    "anxious": "Anxious",
    "frustrated": "Distressed",
    "panicked": "Panicked",
}

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

STAGE1_PROMPT = """You are a board-certified physician with over 15 years of clinical experience \
at a top-tier academic medical center in the United States. You specialize in {department}.

Your task: Generate {n_clusters} clinically realistic and DISTINCT symptom clusters that a real \
patient might present with, appropriate for the {department} department at a "{urgency}" urgency level.

Rules:
1. Each symptom cluster should represent a DIFFERENT clinical scenario (different chief complaint or underlying condition).
2. Include 2-5 symptoms per cluster, listed in order of clinical significance.
3. For each cluster, briefly note the most likely underlying condition (1 sentence).
4. Ensure the urgency level is clinically appropriate — a "{urgency}" case in {department} should genuinely warrant that urgency classification.
5. Do NOT include any treatment recommendations or diagnoses directed at a patient — these are internal clinical notes only.

Respond ONLY with valid JSON. No preamble, no markdown backticks, no explanation.

Format:
{{
  "department": "{department}",
  "urgency": "{urgency}",
  "symptom_clusters": [
    {{
      "cluster_id": 1,
      "symptoms": ["symptom1", "symptom2"],
      "likely_condition": "Brief clinical note"
    }}
  ]
}}"""


STAGE2_PROMPT = """You are generating a realistic patient message for a hospital patient portal \
dataset. The message must read EXACTLY like a real person typed it — not a clinical note, not a \
textbook example, not an AI-generated message. Real patient portal messages are messy, varied, \
and human.

=== CLINICAL CONTEXT (DO NOT reveal any of this in the message) ===
- Department: {department}
- Urgency: {urgency}
- Symptoms to incorporate: {symptoms_list}
- Likely condition (for your reference only): {likely_condition}

=== PATIENT PROFILE — follow each dimension carefully ===

HEALTH LITERACY: {health_literacy}
{health_literacy_guidance}

EMOTIONAL TONE: {emotional_tone}
{emotional_tone_guidance}

WRITING ON BEHALF OF: {on_behalf_of}
{on_behalf_of_guidance}

MESSAGE TYPE: {message_type}
{message_type_guidance}

WRITING STYLE: {writing_style}
{writing_style_guidance}

=== MESSAGE LENGTH ===
{length_guidance}

=== CRITICAL RULES ===
1. Write ONLY the patient's message. Nothing else — no labels, no metadata.
2. Do NOT mention the department name or urgency level anywhere.
3. Do NOT write anything that sounds like a medical textbook or clinical note.
4. Do NOT start with "I am writing to inform you" or similar AI-sounding openers.
5. The health literacy level MUST be clearly reflected in vocabulary, grammar, \
   spelling, and level of medical detail. This is the MOST important dimension.
6. Vary your opening — do not always start with "Hi" or "Hello". Real messages \
   sometimes start mid-thought, with a question, with "So...", with "Hey doc", \
   with no greeting at all, etc.

Respond with ONLY the patient message text."""


# ============================================================================
# DYNAMIC PROMPT GUIDANCE BUILDERS
# ============================================================================

LITERACY_GUIDANCE = {
    "low": """The patient has LOW health literacy. This MUST be clearly visible:
- Uses vague, imprecise language ("something feels off down there", "my chest bone area hurts")
- May use folk terms or incorrect anatomy ("sugar is acting up", "my good kidney")
- Grammar mistakes, run-on sentences, sentence fragments
- May include spelling errors naturally (e.g., "stomache", "diarhea", "perscription")
- Might include irrelevant personal details ("I was at my daughter's soccer game when...")
- May struggle to organize thoughts — message might jump between topics
- Some patients may show ESL (English as Second Language) patterns — awkward phrasing, \
  missing articles, wrong prepositions — this is realistic and encouraged for some messages
- Overall impression: someone who doesn't visit the doctor often and doesn't know \
  the "right" way to describe symptoms""",

    "medium": """The patient has MEDIUM health literacy:
- Generally clear communication but not medically precise
- Knows common medical terms from personal experience (e.g., "blood pressure", \
  "cholesterol", "MRI") but may not use them perfectly
- Mostly correct grammar with occasional informal shortcuts
- Organized enough to get the main point across but may miss relevant details
- Might confuse similar medical concepts or medications""",

    "high": """The patient has HIGH health literacy:
- Uses accurate medical terminology naturally ("intermittent palpitations", \
  "radiating to the left scapula", "paresthesia in the ulnar distribution")
- Well-organized: chief complaint → timeline → associated symptoms → relevant history
- Concise and information-dense — no wasted words
- May reference specific lab values, medications with dosages, or prior procedures
- Reads like someone with medical training or extensive patient experience
- Overall impression: a nurse, a medical student, a chronic disease patient who \
  has learned the vocabulary, or a highly educated professional""",
}

TONE_GUIDANCE = {
    "calm": """CALM: Matter-of-fact, measured phrasing. Not in a rush. May sound almost \
clinical in detachment. Statements rather than questions. No urgency markers.""",

    "anxious": """ANXIOUS: Seeks reassurance. Asks multiple questions ("is this serious?", \
"should I be worried?", "could this be cancer?"). May repeat the same concern in different \
ways. Uses hedging ("I'm probably overthinking this but..."). Overall worried energy.""",

    "frustrated": """FRUSTRATED: Shows impatience or exasperation. References duration or \
failed attempts ("I've been dealing with this for WEEKS", "this is the third time I'm \
writing about this", "nobody seems to take this seriously"). May be curt, demanding, or \
passive-aggressive. Might mention dissatisfaction with prior care.""",

    "panicked": """PANICKED: Urgent, desperate language. May use ALL CAPS for key words, \
multiple exclamation marks, or short frantic sentences ("Please help!!", "I don't know \
what to do", "This is getting worse FAST"). Message may be disorganized because the person \
is typing in a rush. Emotional and raw.""",
}

ON_BEHALF_OF_GUIDANCE = {
    "for_self": """WRITING FOR SELF: First person throughout. "I have been experiencing...", \
"My right knee...", "I noticed that...".""",

    "for_my_child": """WRITING ABOUT THEIR CHILD: The sender is a PARENT writing about their \
son or daughter. MUST include the child's age (pick a realistic age). Use phrases like "my \
4-year-old daughter", "my son who just turned 12", "my baby (8 months)". For young children, \
describe what the parent OBSERVES rather than what the child reports. For older children/teens, \
can include what the child told them. The parent's worry for their child should come through.""",

    "for_my_parent": """WRITING ABOUT THEIR PARENT: The sender is an ADULT CHILD writing about \
their elderly mother or father. MUST include the parent's approximate age. Use phrases like \
"my mom who is 74", "my father (81 years old)", "my dad". May mention the parent's existing \
conditions or medications. May express frustration that the parent won't seek help themselves \
("he refuses to go to the ER", "she keeps saying she's fine but I can tell she's not").""",

    "for_my_spouse": """WRITING ABOUT THEIR SPOUSE/PARTNER: Writing about husband, wife, or \
partner. Use phrases like "my husband", "my wife", "my partner". May include their approximate \
age. Tone may reflect shared concern or frustration ("I've been telling him to see someone \
for months", "she finally agreed to let me message you").""",
}

MESSAGE_TYPE_GUIDANCE = {
    "symptom_report": """SYMPTOM REPORT: Describing symptoms they or someone they know is \
experiencing. This is the most common message type. Should incorporate the listed symptoms \
naturally. Include timeline if the patient profile suggests they'd provide one.""",

    "health_question": """HEALTH QUESTION: A specific, often SHORT question. May have very \
little context. Examples: "Is it normal to have headaches every day?", "Can ibuprofen cause \
stomach bleeding?", "How long should I wait after eating to take my meds?", "Am I overweight \
at 192 lbs for my age (39)?". These are often just 1-2 sentences. Still incorporate at least \
one of the listed symptoms as context for the question.""",

    "follow_up": """FOLLOW-UP: References a previous visit, test, or treatment. Examples: \
"I saw Dr. [name] last week about my knee and it's still not better", "My blood test results \
came back and I have questions", "You prescribed me X two weeks ago and I wanted to update \
you". Should mention some prior interaction with the healthcare system.""",

    "medication_concern": """MEDICATION CONCERN: About a specific drug — side effects, \
interactions, dosing. Examples: "I started metformin 2 weeks ago and I've had diarrhea every \
day since", "Can I take Tylenol with my blood pressure meds?", "I accidentally took a double \
dose of my Lexapro this morning". Should mention at least one medication by name (real, \
common medication names).""",
}

WRITING_STYLE_GUIDANCE = {
    "formal": """FORMAL: Complete sentences, proper punctuation and capitalization, polite \
phrasing. May include "Dear Doctor" or "Thank you for your time". Structured paragraphs.""",

    "casual": """CASUAL: Relaxed but readable. Contractions ("I've", "can't", "it's"), skips \
formalities, may not capitalize perfectly. Conversational tone like texting a friend who \
happens to be a doctor. May use "hey", "so", "anyway" as transitions.""",

    "texting-style": """TEXTING STYLE: Heavy abbreviations (u, ur, pls, bc, rn, idk, w/, b4, \
thx, nvm, tbh, prob, gonna, wanna). Minimal or no punctuation. All lowercase or inconsistent \
caps. Very short sentences or fragments. May use emoji sparingly (but not required). \
Example fragments: "so my knee has been hurting rn for like 2 wks", \
"pls help idk whats wrong w my stomach its been killing me".""",
}


def get_length_guidance(health_literacy: str, message_type: str) -> str:
    """Generate length guidance based on the combination of literacy and message type."""
    if message_type == "health_question":
        return """LENGTH: This is a health question — it MUST be SHORT.
Target: 5-25 words. One sentence, maybe two. Examples of realistic length:
- "is it normal to get headaches every day"
- "can i take advil with my blood pressure meds"
- "how long after antibiotics can i drink alcohol"
- "my son has had a fever for 3 days is that too long to wait"
Do NOT exceed 40 words. Shorter is better and more realistic."""

    if message_type == "medication_concern":
        return """LENGTH: Medication questions are usually SHORT to MEDIUM.
Target: 10-50 words. Get to the point — name the drug, state the concern.
Examples: "I started metformin last week and my stomach has been a mess ever since. Normal?"
Do NOT exceed 80 words."""

    if health_literacy == "low":
        return """LENGTH: Low literacy patients write SHORT or RAMBLING — pick one:
OPTION A — SHORT AND SPARSE (more common, pick this ~60% of the time): 10-35 words. \
Very few details. Example: "my stomach been hurting real bad for 2 days what do i do"
OPTION B — LONG AND UNFOCUSED (~40%): 80-180 words. Includes tangents, irrelevant \
personal details, repeats themselves, hard to follow.
NEVER write a well-organized medium-length message — that's not realistic for low literacy."""

    if health_literacy == "medium":
        return """LENGTH: Medium literacy → MEDIUM length messages.
Target: 25-80 words. Gets the main point across with some context.
This is a normal patient message — not too polished, not too messy.
Do NOT exceed 100 words."""

    # high
    return """LENGTH: High literacy → CONCISE and information-dense.
Target: 30-100 words. Every sentence carries relevant clinical information. \
No filler, no tangents. Think of a nurse or med student messaging a colleague.
May be longer (up to 150 words) ONLY if presenting a complex multi-system issue.
Do NOT pad with unnecessary detail."""


# ============================================================================
# HELPERS
# ============================================================================

def get_urgency_levels(dept): return URGENCY_MAP.get(dept, DEFAULT_URGENCY_LEVELS)

def get_all_combos():
    return [(d, u) for d in DEPARTMENTS for u in get_urgency_levels(d)]

def sample_patient_profile(rng):
    return {
        "health_literacy": rng.choice(HEALTH_LITERACY_OPTIONS),
        "emotional_tone": rng.choice(EMOTIONAL_TONE_OPTIONS),
        "on_behalf_of": rng.choice(ON_BEHALF_OF_OPTIONS),
        "message_type": rng.choice(MESSAGE_TYPE_OPTIONS),
        "writing_style": rng.choice(WRITING_STYLE_OPTIONS),
    }

def parse_json_response(text):
    if not text: return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw (first 300): {text[:300]}")
        return None

def load_json(fp):
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f: return json.load(f)
    return None

def save_json(data, fp):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_stage2_prompt(dept, urg, cluster, profile):
    """Build the full Stage 2 prompt with all dynamic guidance sections."""
    symptoms_str = ", ".join(cluster["symptoms"])
    hl = profile["health_literacy"]
    et = profile["emotional_tone"]
    obo = profile["on_behalf_of"]
    mt = profile["message_type"]
    ws = profile["writing_style"]

    return STAGE2_PROMPT.format(
        department=dept,
        urgency=urg,
        symptoms_list=symptoms_str,
        likely_condition=cluster.get("likely_condition", ""),
        health_literacy=hl,
        health_literacy_guidance=LITERACY_GUIDANCE[hl],
        emotional_tone=et,
        emotional_tone_guidance=TONE_GUIDANCE[et],
        on_behalf_of=obo,
        on_behalf_of_guidance=ON_BEHALF_OF_GUIDANCE[obo],
        message_type=mt,
        message_type_guidance=MESSAGE_TYPE_GUIDANCE[mt],
        writing_style=ws,
        writing_style_guidance=WRITING_STYLE_GUIDANCE[ws],
        length_guidance=get_length_guidance(hl, mt),
    )


# ============================================================================
# API CALL (thread-safe, with retry)
# ============================================================================

def call_openrouter(prompt: str, temperature: float = 0.9) -> Optional[str]:
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MediTriage-LLM",
        "X-OpenRouter-Title": "MediTriage-LLM Data Generator",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content is None:
                print(f"  [Attempt {attempt}/{MAX_RETRIES}] API returned null content")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                    continue
                else:
                    return None
            return content.strip()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] HTTP {status}: {e}")
            if status == 429:
                wait = RETRY_DELAY * attempt * 2
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return None
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return None
    return None


# ============================================================================
# STAGE 1 (synchronous — ~58 calls)
# ============================================================================

def run_stage1():
    print("=" * 60)
    print("STAGE 1: Generating Symptom Clusters")
    print("=" * 60)

    all_clusters = load_json(CLUSTERS_FILE) or []
    existing = {(c["department"], c["urgency"]) for c in all_clusters}
    combos = get_all_combos()
    remaining = [(d, u) for d, u in combos if (d, u) not in existing]

    print(f"Total: {len(combos)} | Done: {len(existing)} | Remaining: {len(remaining)}\n")

    for i, (dept, urg) in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {dept} / {urg}")
        prompt = STAGE1_PROMPT.format(department=dept, urgency=urg, n_clusters=CLUSTERS_PER_COMBO)
        text = call_openrouter(prompt, temperature=0.8)
        parsed = parse_json_response(text)

        if parsed and "symptom_clusters" in parsed:
            parsed["department"] = dept
            parsed["urgency"] = urg
            all_clusters.append(parsed)
            save_json(all_clusters, CLUSTERS_FILE)
            print(f"  ✓ {len(parsed['symptom_clusters'])} clusters")
        else:
            print(f"  ✗ Failed")
            all_clusters.append({"department": dept, "urgency": urg, "symptom_clusters": [], "_error": True})
            save_json(all_clusters, CLUSTERS_FILE)
        time.sleep(1.5)

    total = sum(len(c.get("symptom_clusters", [])) for c in all_clusters)
    failed = sum(1 for c in all_clusters if c.get("_error"))
    print(f"\nDone: {total} clusters. {'⚠ ' + str(failed) + ' failed — re-run to retry' if failed else ''}")


# ============================================================================
# STAGE 2 (concurrent via ThreadPoolExecutor)
# ============================================================================

def run_stage2():
    print("=" * 60)
    print(f"STAGE 2: Generating Patient Messages (model={MODEL}, workers={MAX_WORKERS})")
    print("=" * 60)

    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set."); return

    all_clusters = load_json(CLUSTERS_FILE)
    if not all_clusters:
        print("ERROR: Run Stage 1 first."); return

    dataset = load_json(RAW_DATASET_FILE) or []
    existing_keys = {
        (r["labels"]["department"], r["labels"]["urgency"],
         r["metadata"]["symptom_cluster_id"], r["metadata"]["profile_idx"])
        for r in dataset
    }

    rng = random.Random(RANDOM_SEED)
    tasks = []
    for combo in all_clusters:
        dept, urg = combo["department"], combo["urgency"]
        clusters = combo.get("symptom_clusters", [])
        if not clusters or combo.get("_error"): continue
        for cluster in clusters:
            cid = cluster["cluster_id"]
            for pidx in range(MESSAGES_PER_CLUSTER):
                profile = sample_patient_profile(rng)
                if (dept, urg, cid, pidx) not in existing_keys:
                    tasks.append((dept, urg, cluster, pidx, profile))

    print(f"To generate: {len(tasks)} | Already done: {len(existing_keys)}\n")
    if not tasks:
        print("Nothing to do."); return

    def generate_one(task_idx, dept, urg, cluster, pidx, profile):
        time.sleep(STAGGER_DELAY * (task_idx % MAX_WORKERS))

        cid = cluster["cluster_id"]
        prompt = build_stage2_prompt(dept, urg, cluster, profile)
        text = call_openrouter(prompt, temperature=1.0)

        if text:
            return {
                "patient_message": text,
                "labels": {
                    "department": dept,
                    "urgency": urg,
                    "sentiment": TONE_TO_SENTIMENT[profile["emotional_tone"]],
                },
                "metadata": {
                    **profile,
                    "symptom_cluster_id": cid, "profile_idx": pidx,
                    "symptoms_used": cluster["symptoms"],
                    "likely_condition": cluster.get("likely_condition", ""),
                },
            }
        return None

    BATCH_SIZE = MAX_WORKERS * 3
    total = len(tasks)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = tasks[batch_start : batch_start + BATCH_SIZE]
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(generate_one, batch_start + j, *task): task
                for j, task in enumerate(batch)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    t = futures[future]
                    print(f"  ✗ {t[0]}/{t[1]}/c{t[2]['cluster_id']}/p{t[3]}")

        dataset.extend(results)
        save_json(dataset, RAW_DATASET_FILE)

        progress = min(batch_start + len(batch), total)
        print(f"  [{progress}/{total}] +{len(results)} ok | Total: {len(dataset)}")

    save_json(dataset, RAW_DATASET_FILE)
    print(f"\nStage 2 complete: {len(dataset)} messages")


# ============================================================================
# STAGE 3: POST-PROCESSING
# ============================================================================

def run_stage3():
    print("=" * 60)
    print("STAGE 3: Post-Processing")
    print("=" * 60)

    dataset = load_json(RAW_DATASET_FILE)
    if not dataset:
        print("ERROR: Run Stage 2 first."); return

    print(f"Raw: {len(dataset)}")

    valid_u = {"Low", "Medium", "High", "Emergency"}
    valid_s = {"Calm", "Anxious", "Distressed", "Panicked"}
    valid_d = set(DEPARTMENTS)

    valid, invalid = [], 0
    for r in dataset:
        msg = r.get("patient_message", "").strip()
        lb = r.get("labels", {})
        if msg and len(msg) > 3 and lb.get("department") in valid_d \
           and lb.get("urgency") in valid_u and lb.get("sentiment") in valid_s:
            r["patient_message"] = msg
            valid.append(r)
        else:
            invalid += 1

    print(f"Valid: {len(valid)}" + (f" | Dropped: {invalid}" if invalid else ""))

    seen, deduped = set(), []
    for r in valid:
        key = r["patient_message"].lower().strip()
        if key not in seen:
            seen.add(key); deduped.append(r)

    dups = len(valid) - len(deduped)
    if dups: print(f"Duplicates removed: {dups}")
    print(f"Final: {len(deduped)}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(deduped)

    n = len(deduped)
    t_end, v_end = int(n * 0.70), int(n * 0.85)
    train, val, test = deduped[:t_end], deduped[t_end:v_end], deduped[v_end:]

    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    save_json(train, FINAL_DATASET_DIR / "train.json")
    save_json(val, FINAL_DATASET_DIR / "val.json")
    save_json(test, FINAL_DATASET_DIR / "test.json")
    save_json(deduped, FINAL_DATASET_DIR / "full_dataset.json")

    for label_name, label_key, cats in [
        ("Department", "department", DEPARTMENTS),
        ("Urgency", "urgency", ["Low", "Medium", "High", "Emergency"]),
        ("Sentiment", "sentiment", ["Calm", "Anxious", "Distressed", "Panicked"]),
    ]:
        print(f"\n--- {label_name} Distribution ---")
        counts = {}
        for r in deduped:
            v = r["labels"][label_key]; counts[v] = counts.get(v, 0) + 1
        for c in cats:
            print(f"  {c:<45} {counts.get(c, 0):>4}")

    print("\nStage 3 complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MediTriage-LLM Dataset Generator v2")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == 1: run_stage1()
    elif args.stage == 2: run_stage2()
    elif args.stage == 3: run_stage3()

if __name__ == "__main__":
    main()