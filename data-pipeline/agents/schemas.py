"""
Structured output contracts for the six agents. Kept as Pydantic models so each
CrewAI Task can set output_pydantic=<Model> and get back a typed, validated object
instead of parsing free text -- same spirit as the existing pipeline's strict JSONL
schema (see deterministic_checks.py's ALLOWED_CATEGORIES / ALLOWED_LANGS).
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

ALLOWED_CATEGORIES = ("formality_shift", "sarcasm", "idiom", "emotional_subtext", "none")
Category = Literal["formality_shift", "sarcasm", "idiom", "emotional_subtext", "none"]
Lang = Literal["ru", "en"]


class DetectionVerdict(BaseModel):
    """Detection Agent output -- a cheap gate, not a nuance verdict. Mirrors the
    role deterministic_checks.py plays for Stage A output, but applied to raw
    candidate source sentences (mined or hand-seeded) before they're worth
    spending Annotator Agent attention on."""

    source_text: str
    source_lang: Lang
    worth_annotating: bool = Field(
        description="False for sentences a rule-based tool can already tell carry no "
                    "translatable nuance (no address pronoun, no idiom-lexicon match, "
                    "no known irony cue) -- filters obvious 'none' candidates cheaply, "
                    "same spirit as deterministic_checks.py's free pre-filter."
    )
    candidate_categories: list[Category] = Field(
        default_factory=list,
        description="Categories a rule-based tool found *some* signal for (e.g. "
                    "['formality_shift'] if an address pronoun was found). Not a verdict --"
                    " Annotator Agents still decide has_subtext/category independently.",
    )
    tool_evidence: list[str] = Field(
        default_factory=list,
        description="Short strings naming which tool fired and what it found, e.g. "
                    "'ru_address_pronoun: тебе (2per sing datv)' or "
                    "'idiom_lexicon: какого чёрта -> what the hell'.",
    )


class AnnotationResult(BaseModel):
    """One Annotator Agent's independent judgment on one sentence. Same fields as
    an existing stage_a_batchN.jsonl candidate row, so outputs can be written to
    the same JSONL format and merged with legacy Stage A data."""

    source_lang: Lang
    source_text: str
    translation: str
    has_subtext: bool
    category: Category
    nuance_note: str = Field(
        description="Empty string when has_subtext is false. Under ~20 words when true.",
    )


class AdjudicationResult(BaseModel):
    """Adjudicator Agent output -- same job as stage_b_prefilter.py, as an agent
    with an agreement-calculator tool instead of a single polish/pick API call."""

    source_lang: Lang
    source_text: str
    translation: str
    has_subtext: bool
    category: Category
    nuance_note: str
    resolution: Literal["unanimous", "majority_vote", "escalated_needs_human"]
    agreement_ratio: float = Field(ge=0.0, le=1.0)
    dissenting_agents: list[str] = Field(default_factory=list)


class SourcedCandidate(BaseModel):
    """Sourcing Agent output -- a mined RU-EN pair with a human's own commentary
    on the nuance, found rather than generated. No LLM judgment attached; this is
    raw material for the Detection -> Annotator -> Adjudicator chain, same as a
    hand-written seed line is today."""

    source_lang: Lang
    source_text: str
    human_commentary: Optional[str] = Field(
        default=None,
        description="The bilingual human's own explanation, when the source had one "
                    "(a translator's footnote, a forum answer) -- carried through as a "
                    "reference for the Annotator Agents, not auto-trusted as ground truth.",
    )
    source_url: str
    source_kind: Literal["forum_qa", "translator_note", "subtitle_note", "other"]


class GuidelineProposal(BaseModel):
    """Guideline Agent output -- a proposed addition to config/calibration_guidelines.md,
    surfaced for Akshay's approval rather than auto-applied (mirrors how this session's
    calibration rules were always stated before being applied, never silently assumed)."""

    pattern_description: str = Field(description="What kept getting disagreed on.")
    proposed_rule: str = Field(description="The calibration rule text, written the same "
                                            "way as the existing rules in "
                                            "config/calibration_guidelines.md.")
    direction: Literal["true_to_false", "false_to_true", "clarify_only"]
    supporting_examples: list[str] = Field(
        default_factory=list,
        description="2-4 source_text examples the pattern was drawn from.",
    )
    category: Category


class CoverageReport(BaseModel):
    """Auditor Agent output -- dataset-wide scan, same role as models.json's
    stage_c_coverage_model."""

    total_rows: int
    category_counts: dict[str, int]
    has_subtext_true_ratio: float
    near_duplicate_clusters: list[list[str]] = Field(
        default_factory=list,
        description="Groups of source_text values that are near-duplicates of each "
                    "other (same construction, swapped nouns/names) -- candidates to "
                    "stop generating more of.",
    )
    underrepresented_patterns: list[str] = Field(
        default_factory=list,
        description="Named gaps, e.g. 'no emotional_subtext rows involving apology' -- "
                    "feeds directly into the next batch's seed generation prompt.",
    )
    recommendation: str
