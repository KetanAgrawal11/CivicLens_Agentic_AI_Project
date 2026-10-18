# CivicLense — Change Log

Simple notes on every code change made by Claude.

---

## 2026-10-18

- **Setup:** Created a Python 3.12 virtual environment (`.venv`) and installed all packages from `requirements.txt`. No code changed.

- **`tools.py` — Fixed the crash "unexpected keyword argument 'min_score'".**
  The knowledge-base search tool did not accept the relevance threshold the agent was sending, so every research run failed. It now accepts it and passes it through.

- **`router.py` — Router now checks your documents before choosing "WEB".**
  Before, questions containing words like "protest", "court" or "election" always went to web search, even when matching PDFs were in Qdrant. Now, if your documents match the question well (score ≥ 0.40), it uses **RAG + WEB** (your documents plus a web cross-check).

- **`agent.py` — Agent runs one quick Qdrant search before routing.**
  It gets the best match score for the question and gives it to the router, so the router knows whether your documents are relevant.

- **`agent.py` — Smarter check for when to search research papers (arXiv).**
  Before, the word "ai" hidden inside words like "remain", "said" or "air" wrongly triggered a paper search. Now it matches real words and their forms (e.g. "research" → "researchers", "study" → "studies"), and "AI" only counts as a separate word.

- **Knowledge base cleanup (Qdrant Cloud + `data/metadata.json`).**
  Removed 8 leftover test vectors ("Test ECI Document", "Dup Test") and 2 fake catalog entries ("Climate Accord 2025", "Statute 1950"). The ECI SIR Order's text was saved under the wrong name "Source 1", so it was renamed back, and the duplicate file `Source 1.pdf` was deleted. Now: 5 real documents, 1,066 vectors.

- **`knowledge_base.py` — Same file can no longer be uploaded twice under a different name.**
  Before, uploading an identical file with a new name silently overwrote the first document's data. Now it is skipped with a message saying which document already has that content.

- **`tests/conftest.py` (new) + `tests/test_scenarios.py` — Tests never touch your real database.**
  All tests now use a temporary in-memory database, so running tests can't add junk to Qdrant Cloud anymore.

- **`tests/test_knowledge_base.py` — Fixed a failing test.**
  It searched with a 3-word query that scored just below the 0.40 relevance cut-off; now it uses a full question. All 50 tests pass.

- **`agent.py` — The AI planner now decides which extra tools run.**
  Wikipedia and research-paper (arXiv) searches run only when the planner picks them. The old keyword rules are used only as a backup if the planner fails. Unknown tool names from the planner are ignored.

- **`app.py` + `agent.py` — The Routing Strategy dropdown and both sliders now work.**
  Before, they were ignored. Now you can force a mode (documents only, web only, both, or direct), and "Reasoning Depth" and "Passage Retrieval Depth" really control how many search rounds and document pieces are used.

- **`app.py` — Live progress bar and research plan now update during a run.**
  The progress function received its inputs in the wrong order and used wrong step names, so it silently failed.

- **`app.py` — "Submit Follow-up" now actually runs the follow-up question.**
  Before, it only copied the text into the question box.

- **`tools.py` + `app.py` — Replaced the fake "85% Engine Confidence" with real "Evidence Coverage".**
  85% was a hard-coded number. Now it shows how many claims were matched to real sources (e.g. "6 of 8 claims"), or "N/A" when no claims were checked.

- **`app.py` — The Evidence & Claims Matrix now shows each claim's real verdict and evidence.**
  It was reading data fields that never existed, so it only showed a plain list.

- **`app.py` + `.streamlit/config.toml` — New professional UI design.**
  Clean light theme (Inter font, one blue accent), simple line icons instead of emojis, no hype words or fake "Active Cluster" light. The sidebar is gone: the question and run settings sit side by side, and results use the full width. Status chips at the top show the real document count and model.

- **`app.py` — Fixed the example question buttons loading the wrong question.**
  E.g. "Supreme Court Ruling" loaded the renewable-energy question. Replaced the 4 buttons with one example picker that loads the correct text.

- **`app.py` — Results are easier to read.**
  Live progress now appears in a collapsible status box; the activity log is a sortable table; the document list is a table with a safe "Delete" pop-up; text from websites is escaped so it can't break the page layout.

- **`app.py` — Fixed the logo and status bar being hidden at the top.**
  The page's top spacing was smaller than Streamlit's fixed toolbar ("Deploy" bar), so the header slid underneath it. Added enough space so it is fully visible.

- **`app.py` + `.streamlit/config.toml` — Went back to the original layout, made professional.**
  You preferred the original design, so the original layout is back (dark header bar, sidebar, coloured verdict card, dark activity terminal), but polished: all emojis replaced with clean line icons, hype words removed ("ENTERPRISE", fake "Active Cluster"), and the header now shows real status (Ready / Setup needed, document and chunk count, model). All earlier fixes are kept.

- **`app.py` — The 5-step progress tracker now updates live.**
  Before, the steps were drawn once and never changed. Now each step turns blue while running and green with a tick when done.

- **`app.py` — Black execution-log terminal now stays visible after a run.**
  It always showed during a run, but disappeared when results loaded (it was only inside the "Agent Execution Trace" tab). Now it also appears on the main results page, under the metrics.

- **`agent.py` + `tools.py` — Fixed false "Contradicted" verdicts; the AI now judges each claim.**
  Before, a claim counted as "contradicted" if any related web page contained words like "rejected" or "failed" anywhere, so plain facts (e.g. "ECI issued the order on 24 June 2025") were marked contradicted. Now, in the same AI call that extracts claims, each claim is judged against numbered sources with strict rules (only an explicit opposite statement counts as a contradiction), and the reason is shown. The old word check stays as a backup and now only looks at sentences about the claim.

- **`tools.py` — Fairer overall verdict.**
  One disputed claim no longer makes the whole result "Contradicted"; that now needs contradicted claims to be at least as many as supported ones.

- **`agent.py` + `config.py` — "Reasoning depth" can now really run more than 1 round.**
  Page downloads no longer use up the 12-tool-call budget, each follow-up round gets its own quota of 4 new web pages, and document passages from earlier rounds are kept instead of replaced. The results line now shows rounds used, pages downloaded and why the agent stopped.

- **`agent.py` — Groq "rate limit" no longer kills the run.**
  When Groq's per-minute limit is hit, the agent waits the time Groq asks for (max 30s) and retries up to 3 times, showing the wait in the log.

- **`tests/test_verdicts.py` (new) — 7 tests for the new verdict logic and retry.** All 57 tests pass.

- **`agent.py` — Fixed strange citations like "【WEB-1†L1-L4】" in reports.**
  The Groq model (gpt-oss) sometimes writes citations in its own built-in style with made-up line numbers. The report prompt now forbids that style, and any that still appear are automatically converted to the normal `[WEB-1]` form.

- **`app.py` — Live Tool Tester now has all 11 tools (was 5).**
  Added fetch_webpage, search_knowledge_base, summarize_document, extract_claims, compare_evidence and verify_citations. Each tool has its own input fields with a working example (text box, document picker, or editable JSON), and bad JSON shows a clear warning instead of crashing.

- **`session_store.py` (new) + `app.py` + `config.py` — Sidebar now shows your research sessions.**
  Every finished run is saved to `data/sessions/` (not uploaded to git). The sidebar lists the current session's questions and your past sessions; clicking one reopens its results and restores its memory so follow-ups still work. "New session" now keeps the old session instead of deleting it, and sessions can be deleted from the sidebar. Added 3 tests (62 pass).

- **`report_export.py` (new) + `app.py` — One "Export" button with Markdown, JSON and PDF.**
  The two export buttons are replaced by a single "Export" menu. The new PDF contains a summary table plus all 3 tabs: Verified Report (with headings, lists and tables), Evidence & Claims Matrix (colour-coded verdicts with reasons) and Sources & Credibility (clickable links), with page numbers. Special characters the font can't draw (e.g. the AI's non-breaking hyphens) are fixed so no empty boxes appear. Added `reportlab` and `markdown` to requirements and 2 tests (64 pass).

- **`app.py` + `agent.py` — Black terminal is now always visible, even at the very start.**
  Before, it stayed hidden during "Route & plan" because the first log line only arrived after the embedding model loaded and Qdrant was searched. Now an idle terminal shows before any run, a "Starting…" line appears the moment you click Execute, and the agent logs right at the start of routing. The terminal also sits above the research plan so it doesn't jump down.

- **`app.py` — Log times now show your local time (IST) instead of UTC.**
  Times are still saved in UTC (standard practice), but the execution log and the document "Added" dates are now converted to this computer's time zone, e.g. 16:29 UTC now shows as 21:59.

- **`app.py` — Quick example buttons now load the correct question.**
  The 4 buttons are kept, but each now loads its matching example and also updates the sidebar preset.
