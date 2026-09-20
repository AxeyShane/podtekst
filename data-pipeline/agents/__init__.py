"""
Podtekst's six-agent architecture (CrewAI-based), added 2026-09-20.

Each role maps to a source file. This package is an EXPERIMENTAL PARALLEL PATH
next to the proven run_batch.py pipeline (stage_a_generate.py + stage_b_prefilter.py
+ Cowork verification) -- it does not replace it. See ../AGENTS.md for the full
role-to-file map and current status of each agent.

    Sourcing Agent    -> agents_source.py     (mines RU-EN parallel text + human
                                                commentary from forums)
    Detection Agent   -> agents/tools.py's flag_* tools, called from agents_annotate.py
                                               (cheap rule-based gate: is this sentence
                                                worth spending Annotator attention on)
    Annotator Agents  -> agents/agent_defs.py's build_annotator_agents()
                                               (one CrewAI Agent per active
                                                config/models.json stage_a_generators
                                                entry -- same 4-model roster as
                                                stage_a_generate.py, wrapped as agents
                                                with tool access instead of one-shot calls)
    Adjudicator Agent -> agents_annotate.py    (agreement check across Annotators;
                                                auto-resolves unanimous, escalates the rest --
                                                same job as stage_b_prefilter.py, as an agent)
    Guideline Agent   -> agents_guideline.py   (reads disagreement patterns, proposes
                                                new rows for config/calibration_guidelines.md)
    Auditor Agent     -> agents_audit.py       (dataset-wide coverage/redundancy scan,
                                                same role as models.json's
                                                stage_c_coverage_model)
"""
