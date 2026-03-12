"""
MediTriage-LLM: Patient-Perspective Message Generator
=====================================================
Generates realistic patient portal messages WITHOUT pre-assigned departments
or symptom clusters. Messages are generated purely from the patient's
perspective with randomized patient profiles.

Labels (department, urgency) will be annotated separately by a teammate
using a stronger model — this script only generates raw messages + metadata.

Usage:
    pip install requests
    export OPENROUTER_API_KEY="your-key-here"
    python generate_patient_messages.py
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

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# --- Model Selection ---
# Free:  "stepfun/step-3.5-flash:free"        → $0 total
# Cheap: "google/gemini-2.0-flash-001"         → ~$0.60 for 3000 msgs
# Cheap: "openai/gpt-4.1-nano"                 → ~$0.60 for 3000 msgs
# Good:  "openai/gpt-4.1-mini"                 → ~$2.50 for 3000 msgs
MODEL = "openai/gpt-4.1-nano"

MAX_RETRIES = 3
RETRY_DELAY = 5

# --- Concurrency ---
MAX_WORKERS = 4
STAGGER_DELAY = 0.3

# --- Output ---
OUTPUT_DIR = Path("data/raw")
RAW_FILE = OUTPUT_DIR / "patient_messages.json"

# --- Generation ---
TOTAL_MESSAGES = 3000
RANDOM_SEED = 42

# ============================================================================
# PATIENT PROFILE DIMENSIONS
# ============================================================================

HEALTH_LITERACY_OPTIONS = ["low", "medium", "high"]

EMOTIONAL_TONE_OPTIONS = [
    "calm",
    "anxious",
    "frustrated",
    "panicked",
    "curious",
    "surprised",
    "sad",
    "confused",
    "helpless",
    "hopeful",
]

ON_BEHALF_OF_OPTIONS = ["for_self", "for_my_child", "for_my_parent", "for_my_spouse"]

MESSAGE_TYPE_OPTIONS = ["symptom_report", "health_question", "follow_up", "medication_concern"]

WRITING_STYLE_OPTIONS = ["formal", "casual", "texting-style"]

# emotional_tone → sentiment label
TONE_TO_SENTIMENT = {
    "calm": "Calm",
    "anxious": "Anxious",
    "frustrated": "Frustrated",
    "panicked": "Panicked",
    "curious": "Curious",
    "surprised": "Surprised",
    "sad": "Sad",
    "confused": "Confused",
    "helpless": "Helpless",
    "hopeful": "Hopeful",
}

# ============================================================================
# PROMPT
# ============================================================================

PROMPT_TEMPLATE = """You are generating a realistic patient message for a hospital patient portal \
dataset. The message must read EXACTLY like a real person typed it — not a clinical note, not a \
textbook example, not an AI-generated message.

Your goal: Write a message that a REAL patient would send to their doctor through an online \
portal. The patient has some health concern — it could be ANYTHING: a weird symptom, a chronic \
issue, a question about medication, a follow-up from a visit, something about their kid, \
something minor, something scary, something embarrassing. Real life is diverse — be creative \
and realistic. Do NOT default to dramatic or urgent scenarios every time. Many real portal \
messages are mundane, routine, or even trivial.

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
1. Write ONLY the patient's message. Nothing else — no labels, no metadata, no commentary.
2. Do NOT write anything that sounds like a medical textbook or clinical note.
3. Do NOT start with "I am writing to inform you" or similar AI-sounding openers.
4. The health literacy level MUST be clearly reflected in vocabulary, grammar, \
   spelling, and level of medical detail. This is the MOST important dimension.
5. Vary your opening — do not always start with "Hi" or "Hello". Real messages \
   sometimes start mid-thought, with a question, with "So...", with "Hey doc", \
   with no greeting at all, etc.
6. The health concern should feel NATURAL and UNSCRIPTED — as if the patient just \
   decided to message their doctor about whatever is on their mind. Do NOT try to \
   cover multiple organ systems or create a dramatic scenario. Simple, single-issue \
   messages are the most common and realistic.
7. NEVER mention a department name (like "cardiology" or "orthopedics") — real patients \
   don't know or use these terms in their messages.
8. Each message you generate must be COMPLETELY DIFFERENT from any other — vary the \
   health concern, body part, severity, context, and patient situation.

Respond with ONLY the patient message text."""


# ============================================================================
# DYNAMIC GUIDANCE
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

    "curious": """CURIOUS: Genuinely interested, asking questions out of intellectual curiosity \
rather than fear. "I was reading about X and wondered...", "Is it true that...?", "I've always \
wanted to ask — why does my body do X?". Tone is exploratory and engaged, not worried.""",

    "surprised": """SURPRISED: Something unexpected happened and the patient is caught off guard. \
"I never had this before", "This came out of nowhere", "I was shocked when I noticed...". \
Tone is bewildered but not necessarily panicked — more like "what is going on?".""",

    "sad": """SAD: The patient is emotionally down about their health situation. May express \
feeling defeated, tired of dealing with a chronic issue, or grieving a diagnosis. "I'm just \
so tired of this", "It feels like nothing ever gets better", "I don't know how much longer \
I can deal with this". Subdued, low-energy tone — not dramatic, just weary.""",

    "confused": """CONFUSED: The patient doesn't understand something — their diagnosis, their \
medication instructions, test results, or what's happening to their body. "I don't really \
understand what the doctor told me", "The instructions say X but I thought it was Y", "Can \
someone explain this in plain English?". May ask the same thing multiple ways.""",

    "helpless": """HELPLESS: Feels overwhelmed and doesn't know what to do. "I've tried \
everything and nothing works", "I don't know where to turn", "I feel like I'm just getting \
worse". Different from panicked — less frantic, more resigned. May ask for guidance or \
direction rather than immediate help.""",

    "hopeful": """HOPEFUL: Positive or optimistic tone despite health concerns. "I'm feeling \
better since the last visit and wanted to check in", "I heard there's a new treatment — \
could that work for me?", "Things are improving and I wanted to update you". Upbeat but \
still has a genuine health reason for messaging.""",
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
experiencing. This is the most common message type. Include timeline if the patient profile \
suggests they'd provide one.""",

    "health_question": """HEALTH QUESTION: A specific, often SHORT question. May have very \
little context. Examples: "Is it normal to have headaches every day?", "Can ibuprofen cause \
stomach bleeding?", "How long should I wait after eating to take my meds?", "Am I overweight \
at 192 lbs for my age (39)?". These are often just 1-2 sentences.""",

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

def sample_patient_profile(rng):
    return {
        "health_literacy": rng.choice(HEALTH_LITERACY_OPTIONS),
        "emotional_tone": rng.choice(EMOTIONAL_TONE_OPTIONS),
        "on_behalf_of": rng.choice(ON_BEHALF_OF_OPTIONS),
        "message_type": rng.choice(MESSAGE_TYPE_OPTIONS),
        "writing_style": rng.choice(WRITING_STYLE_OPTIONS),
    }


def build_prompt(profile):
    hl = profile["health_literacy"]
    et = profile["emotional_tone"]
    obo = profile["on_behalf_of"]
    mt = profile["message_type"]
    ws = profile["writing_style"]

    return PROMPT_TEMPLATE.format(
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


def load_json(fp):
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(data, fp):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# API CALL
# ============================================================================

def call_openrouter(prompt: str, temperature: float = 1.0) -> Optional[str]:
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
# MAIN GENERATION
# ============================================================================

def run():
    print("=" * 60)
    print(f"Patient Message Generator (model={MODEL}, workers={MAX_WORKERS})")
    print(f"Target: {TOTAL_MESSAGES} messages")
    print("=" * 60)

    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.")
        return

    # Load existing progress
    dataset = load_json(RAW_FILE) or []
    done = len(dataset)
    remaining = TOTAL_MESSAGES - done

    print(f"Already done: {done} | Remaining: {remaining}\n")

    if remaining <= 0:
        print("Nothing to do — target already reached.")
        return

    rng = random.Random(RANDOM_SEED)
    # Advance RNG past already-generated profiles for reproducibility
    for _ in range(done):
        sample_patient_profile(rng)

    # Build task list
    tasks = []
    for i in range(remaining):
        profile = sample_patient_profile(rng)
        msg_id = done + i
        tasks.append((msg_id, profile))

    def generate_one(task_idx, msg_id, profile):
        time.sleep(STAGGER_DELAY * (task_idx % MAX_WORKERS))

        prompt = build_prompt(profile)
        text = call_openrouter(prompt, temperature=1.0)

        if text:
            return {
                "id": msg_id,
                "patient_message": text,
                "labels": {
                    "sentiment": TONE_TO_SENTIMENT[profile["emotional_tone"]],
                    # department and urgency will be annotated by teammate
                },
                "metadata": {
                    **profile,
                },
            }
        return None

    # Process in batches
    BATCH_SIZE = MAX_WORKERS * 3
    total = len(tasks)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = tasks[batch_start: batch_start + BATCH_SIZE]
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(generate_one, batch_start + j, msg_id, profile): (msg_id, profile)
                for j, (msg_id, profile) in enumerate(batch)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    mid, _ = futures[future]
                    print(f"  ✗ Failed msg_id={mid}")

        # Sort by id to maintain order
        results.sort(key=lambda r: r["id"])
        dataset.extend(results)
        save_json(dataset, RAW_FILE)

        progress = min(batch_start + len(batch), total)
        print(f"  [{done + progress}/{TOTAL_MESSAGES}] +{len(results)} ok | Total: {len(dataset)}")

    save_json(dataset, RAW_FILE)

    # --- Quick stats ---
    print(f"\nGeneration complete: {len(dataset)} messages")
    print(f"Saved to: {RAW_FILE}")

    # Dedup check
    seen = set()
    dups = 0
    for r in dataset:
        key = r["patient_message"].lower().strip()
        if key in seen:
            dups += 1
        seen.add(key)
    if dups:
        print(f"⚠ {dups} duplicates detected (will be cleaned during annotation)")

    # Profile distribution
    for dim in ["health_literacy", "emotional_tone", "on_behalf_of", "message_type", "writing_style"]:
        counts = {}
        for r in dataset:
            v = r["metadata"][dim]
            counts[v] = counts.get(v, 0) + 1
        print(f"\n--- {dim} ---")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {k:<25} {v:>4} ({100*v/len(dataset):.1f}%)")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run()