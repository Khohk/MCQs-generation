"""
app.py — MCQ Generator Streamlit UI
4 tabs: Upload → Processing → Review → Export
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import streamlit as st

st.set_page_config(
    page_title="MCQ Generator",
    page_icon="=",
    layout="wide",
    initial_sidebar_state="collapsed",
)

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

_defaults = {
    "pdf_path": None, "pdf_name": None,
    "pages": None, "chunks": None,
    "raw_mcqs": None, "mcqs": None, "stats": None,
    "config": {},
    "page_range": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.markdown(
    "<h1 style='text-align:center;color:#1E3A5F'>MCQ Generator</h1>"
    "<p style='text-align:center;color:#6C757D'>Tu dong sinh cau hoi trac nghiem tu slide PDF</p>",
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Processing", "Review", "Export"])

# ══ TAB 1 — UPLOAD ════════════════════════════════════════════════
with tab1:
    st.subheader("Tai len PDF va cau hinh")
    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded = st.file_uploader("Chon file PDF slide", type=["pdf"])
        if uploaded is not None:
            save_path = UPLOAD_DIR / uploaded.name
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.pdf_path = str(save_path.resolve())
            st.session_state.pdf_name = uploaded.name
            # Reset kết quả cũ khi upload file mới
            for k in ["pages", "chunks", "raw_mcqs", "mcqs", "stats", "page_range"]:
                st.session_state[k] = None
            st.success(f"File: **{uploaded.name}** ({uploaded.size // 1024} KB)")

        if st.session_state.pdf_name:
            st.info(f"Dang dung: {st.session_state.pdf_name}")

            # ── Checkpoint info ──────────────────────────────────
            from pipeline.generator import get_checkpoint_info
            ckpt = get_checkpoint_info(st.session_state.pdf_name)
            if ckpt:
                st.warning(
                    f"Co checkpoint cu: {ckpt['chunks_done']} chunks, "
                    f"{ckpt['mcqs_so_far']} MCQs. "
                    f"Se **resume** tu diem dung. Nhan 'Xoa checkpoint' de chay lai tu dau."
                )
                if st.button("Xoa checkpoint, chay lai tu dau"):
                    from pipeline.generator import clear_checkpoint
                    clear_checkpoint(st.session_state.pdf_name)
                    st.rerun()

    with col2:
        st.markdown("**Cau hinh**")
        n_q   = st.slider("So cau/chunk", 1, 5, 2)
        diff  = st.selectbox("Do kho", ["easy", "medium", "hard"], index=1)
        strat = st.selectbox("Chunking", ["auto", "title", "fixed"], index=0)
        maxp  = st.slider("Max trang/chunk", 2, 10, 6)

        st.markdown("**Chon range trang (optional)**")
        use_range = st.checkbox("Chi xu ly 1 phan PDF", value=False)
        page_from, page_to = 1, 999
        if use_range:
            col_a, col_b = st.columns(2)
            with col_a:
                page_from = st.number_input("Tu trang", min_value=1, value=1)
            with col_b:
                page_to   = st.number_input("Den trang", min_value=1, value=50)

        st.session_state.config = {
            "n_questions": n_q,
            "difficulty":  diff,
            "strategy":    strat,
            "max_pages":   maxp,
            "page_from":   page_from if use_range else None,
            "page_to":     page_to   if use_range else None,
        }

    st.divider()
    if st.session_state.pdf_name:
        if st.button("Bat dau xu ly", type="primary", use_container_width=True):
            for k in ["pages", "chunks", "raw_mcqs", "mcqs", "stats"]:
                st.session_state[k] = None
            st.rerun()
    else:
        st.info("Upload PDF truoc")

# ══ TAB 2 — PROCESSING ════════════════════════════════════════════
with tab2:
    st.subheader("Xu ly Pipeline")

    has_pdf    = bool(st.session_state.pdf_path and Path(st.session_state.pdf_path).exists())
    has_result = st.session_state.mcqs is not None
    run_now    = has_pdf and not has_result

    if st.button("Chay lai", disabled=not has_pdf):
        for k in ["pages", "chunks", "raw_mcqs", "mcqs", "stats"]:
            st.session_state[k] = None
        run_now = True

    if run_now:
        cfg      = st.session_state.config
        pdf_path = st.session_state.pdf_path
        pdf_name = st.session_state.pdf_name
        log_area = st.empty()
        prog     = st.progress(0, text="Bat dau...")
        logs: list = []

        def log(msg: str):
            safe = msg.encode("ascii", errors="replace").decode("ascii")
            logs.append(safe)
            log_area.code("\n".join(logs[-30:]))

        # Step 1: Parse
        try:
            prog.progress(5, text="[1/4] Parsing PDF...")
            log(f"[1/4] Parsing: {pdf_name}")
            from pipeline.pdf_parser import parse_pdf, get_pdf_metadata
            meta  = get_pdf_metadata(pdf_path)
            pages = parse_pdf(pdf_path)

            # Apply page range filter
            pf, pt = cfg.get("page_from"), cfg.get("page_to")
            if pf and pt:
                pages = [p for p in pages if pf <= p["page_num"] <= pt]
                log(f"      -> Range filter: trang {pf}-{pt}")

            st.session_state.pages = pages
            log(f"      -> {len(pages)}/{meta['total_pages']} pages extracted")
            prog.progress(25, text="[1/4] Parse xong")
        except Exception as e:
            st.error(f"Loi parse PDF: {e}")
            st.stop()

        # Step 2: Chunk
        try:
            prog.progress(30, text="[2/4] Chunking...")
            log(f"[2/4] Chunking (strategy={cfg.get('strategy','auto')})...")
            from pipeline.chunker import chunk_pages
            chunks = chunk_pages(pages,
                strategy=cfg.get("strategy", "auto"),
                max_pages=cfg.get("max_pages", 6))
            st.session_state.chunks = chunks

            # Hiển thị ETA
            from pipeline.generator import estimate_seconds, get_checkpoint_info
            ckpt = get_checkpoint_info(pdf_name)
            n_done = ckpt["chunks_done"] if ckpt else 0
            eta = estimate_seconds(len(chunks), n_done)
            log(f"      -> {len(chunks)} chunks | ETA: ~{eta}s (~{eta//60}m{eta%60}s)")
            if n_done > 0:
                log(f"      -> Resume: {n_done} chunks da co checkpoint, bo qua")
            prog.progress(45, text="[2/4] Chunk xong")
        except Exception as e:
            st.error(f"Loi chunking: {e}")
            st.stop()

        # Step 3: Generate
        try:
            prog.progress(50, text="[3/4] Goi Gemini API...")
            log(f"[3/4] Generating ({cfg.get('n_questions',2)}/chunk, {cfg.get('difficulty','medium')})...")
            from pipeline.generator import generate_mcqs
            gen_prog = st.progress(0, text="Gemini: 0%")

            def on_progress(idx, total, chunk_id, eta_sec, skipped=False):
                pct  = int(idx / total * 100)
                mins = eta_sec // 60
                secs = eta_sec % 60
                label = f"[skip]" if skipped else f"ETA {mins}m{secs:02d}s"
                gen_prog.progress(pct, text=f"Gemini: {chunk_id} ({idx+1}/{total}) {label}")
                status = "skip" if skipped else "..."
                log(f"      [{idx+1}/{total}] {chunk_id} {status}")

            raw_mcqs = generate_mcqs(
                chunks,
                n_per_chunk=cfg.get("n_questions", 2),
                difficulty=cfg.get("difficulty", "medium"),
                pdf_name=pdf_name,
                on_progress=on_progress,
            )
            st.session_state.raw_mcqs = raw_mcqs
            gen_prog.progress(100, text="Gemini: done")
            log(f"      -> {len(raw_mcqs)} raw MCQs")
            prog.progress(80, text="[3/4] Generate xong")
        except Exception as e:
            st.error(f"Loi generate: {e}")
            st.stop()

        # Step 4: Validate
        try:
            prog.progress(85, text="[4/4] Validating...")
            log("[4/4] Validating...")
            from pipeline.validator import validate_mcqs, validation_stats
            valid, rejected = validate_mcqs(raw_mcqs)
            stats = validation_stats(valid, rejected)
            st.session_state.mcqs  = valid
            st.session_state.stats = stats
            log(f"      -> Valid: {stats['valid']} | Rejected: {stats['rejected']} | {stats['pass_rate']}%")
            prog.progress(100, text="Hoan tat!")
        except Exception as e:
            st.error(f"Loi validate: {e}")
            st.stop()

    # Kết quả
    if st.session_state.mcqs is not None:
        st.success("Pipeline hoan tat!")
        stats = st.session_state.stats or {}
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pages",      len(st.session_state.pages  or []))
        c2.metric("Chunks",     len(st.session_state.chunks or []))
        c3.metric("MCQs valid", stats.get("valid", 0))
        c4.metric("Pass rate",  f"{stats.get('pass_rate', 0)}%")
        mm = stats.get("bloom_mismatch_count", 0)
        c5.metric("Bloom mismatch", mm, delta=f"{stats.get('bloom_mismatch_rate',0)}%",
                  delta_color="inverse" if mm > 0 else "off")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Bloom distribution**")
            st.json(stats.get("bloom_dist", {}))
        with cb:
            st.markdown("**Difficulty distribution**")
            st.json(stats.get("difficulty_dist", {}))
        if stats.get("reject_reasons"):
            with st.expander("Rejected reasons"):
                st.json(stats["reject_reasons"])
    elif not has_pdf:
        st.info("Upload PDF o Tab 1 truoc")

# ══ TAB 3 — REVIEW ════════════════════════════════════════════════
with tab3:
    st.subheader("Review & Chinh sua MCQ")
    mcqs = st.session_state.get("mcqs")

    if not mcqs:
        st.info("Chua co MCQ. Chay pipeline o Tab 2 truoc.")
    else:
        c1, c2, c3 = st.columns(3)
        all_blooms = sorted(set(m["bloom_level"] for m in mcqs))
        all_diffs  = sorted(set(m["difficulty"]  for m in mcqs))
        all_chunks = sorted(set(m["source_chunk"] for m in mcqs))
        with c1: sel_b = st.multiselect("Bloom", all_blooms, default=all_blooms)
        with c2: sel_d = st.multiselect("Difficulty", all_diffs, default=all_diffs)
        with c3: sel_c = st.multiselect("Chunk", all_chunks, default=all_chunks)

        filtered = [m for m in mcqs
                    if m["bloom_level"] in sel_b
                    and m["difficulty"] in sel_d
                    and m["source_chunk"] in sel_c]
        st.caption(f"Hien thi {len(filtered)} / {len(mcqs)} MCQs")
        st.divider()

        to_delete = []
        for i, mcq in enumerate(filtered):
            label = f"Q{i+1} [{mcq['bloom_level']}|{mcq['difficulty']}] {mcq['question'][:70]}"
            with st.expander(label, expanded=(i == 0)):
                mcq["question"] = st.text_area("Question", value=mcq["question"], key=f"q{i}")
                # Lay answer hien tai tu session (co the da duoc sua)
                cur_ans = mcq.get("answer", "A")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    cur_ans = st.selectbox("Correct answer", ["A","B","C","D"],
                        index=["A","B","C","D"].index(cur_ans), key=f"ans{i}")
                    mcq["answer"] = cur_ans

                # Options hien thi sau khi biet cur_ans → marker [ans] luon dung
                oc1, oc2 = st.columns(2)
                for j, opt in enumerate(["A","B","C","D"]):
                    marker = "  [ans]" if opt == cur_ans else ""
                    with (oc1 if j % 2 == 0 else oc2):
                        mcq[opt] = st.text_input(f"{opt}{marker}", value=mcq[opt], key=f"o{opt}{i}")
                with mc2:
                    blooms = ["remember","understand","apply","analyze","evaluate","create"]
                    mcq["bloom_level"] = st.selectbox("Bloom level", blooms,
                        index=blooms.index(mcq.get("bloom_level","understand")), key=f"bl{i}")
                with mc3:
                    diffs = ["easy","medium","hard"]
                    mcq["difficulty"] = st.selectbox("Difficulty", diffs,
                        index=diffs.index(mcq.get("difficulty","medium")), key=f"df{i}")

                # Warn neu bloom khong khop difficulty
                bloom_map = {"easy":{"remember","understand"}, "medium":{"apply","analyze"}, "hard":{"evaluate","create"}}
                if mcq["bloom_level"] not in bloom_map.get(mcq["difficulty"], set()):
                    st.warning(f"Bloom/Difficulty mismatch: {mcq['bloom_level']} khong phu hop voi {mcq['difficulty']}")
                mcq["explanation"] = st.text_area("Explanation", value=mcq["explanation"], key=f"ex{i}")
                if st.button("Xoa MCQ nay", key=f"del{i}"):
                    to_delete.append(i)

        if to_delete:
            st.session_state.mcqs = [m for idx, m in enumerate(filtered) if idx not in to_delete]
            st.rerun()

# ══ TAB 4 — EXPORT ════════════════════════════════════════════════
with tab4:
    st.subheader("Export MCQ")
    mcqs = st.session_state.get("mcqs")

    if not mcqs:
        st.info("Chua co MCQ. Chay pipeline o Tab 2 truoc.")
    else:
        st.success(f"San sang export {len(mcqs)} MCQs")
        fname = (st.session_state.get("pdf_name") or "mcq").replace(".pdf", "")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Kahoot (.xlsx)**")
            if st.button("Tao Kahoot", use_container_width=True):
                try:
                    from pipeline.exporter import export_kahoot
                    st.download_button("Download Kahoot.xlsx",
                        data=export_kahoot(mcqs),
                        file_name=f"kahoot_{fname}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)
                except Exception as e:
                    st.error(f"Loi: {e}")

        with c2:
            st.markdown("**Quizizz (.csv)**")
            if st.button("Tao Quizizz", use_container_width=True):
                try:
                    from pipeline.exporter import export_quizizz
                    st.download_button("Download Quizizz.csv",
                        data=export_quizizz(mcqs),
                        file_name=f"quizizz_{fname}.csv",
                        mime="text/csv",
                        use_container_width=True)
                except Exception as e:
                    st.error(f"Loi: {e}")

        with c3:
            st.markdown("**JSON (raw)**")
            st.download_button("Download MCQs.json",
                data=json.dumps(mcqs, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"mcqs_{fname}.json",
                mime="application/json",
                use_container_width=True)

        # Xóa checkpoint sau khi export thành công
        if st.button("Xoa checkpoint sau khi export xong", use_container_width=True):
            from pipeline.generator import clear_checkpoint
            clear_checkpoint(st.session_state.get("pdf_name", ""))
            st.success("Da xoa checkpoint.")

        st.divider()
        import pandas as pd
        st.dataframe(pd.DataFrame([{
            "Question":   m["question"][:60]+"...",
            "Answer":     m["answer"],
            "Bloom":      m["bloom_level"],
            "Difficulty": m["difficulty"],
            "Chunk":      m["source_chunk"],
        } for m in mcqs]), use_container_width=True, height=300)