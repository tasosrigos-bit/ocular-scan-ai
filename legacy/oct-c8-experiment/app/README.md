# Clinic Case Browser

A chat-style Streamlit viewer for the **pre-computed** Part 2 narratives. It runs
no model and needs no GPU — it reads only the files `rag_narrate.py` already
wrote:

```
<output_dir>/rag/narratives_<split>.jsonl
<output_dir>/rag/summary_<split>.json
```

For each of the 37 clinic cases it shows the OCT scan, Part 1's frozen decision
(predicted class, confidence, urgency badge, differential), and then — matching
the system's real routing — either the citation-grounded narrative, or a "skipped
(Normal / abstained)" notice, or a "failed verification → Part 1 template"
notice. The **Verified** indicator reads straight from `narrator_meta.verified`.

## Run it locally

```bash
pip install -e ".[app]"
python rag_narrate.py paths=default rag_run.split=external_test    # once, to make the artifacts
streamlit run app/case_browser.py
```

The artifacts directory defaults to the Kaggle path; set it in the sidebar or via
`OCT_CDS_RAG_DIR=/abs/path/to/<output_dir>/rag`.

## Run it on Kaggle

Kaggle notebooks don't serve web apps directly. Tunnel the Streamlit port with
**cloudflared** (no account, no token). In one cell, after `rag_narrate.py` has
produced the artifacts:

```python
!pip install -q streamlit
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /tmp/cloudflared && chmod +x /tmp/cloudflared

import subprocess, time, re, os
os.environ["OCT_CDS_RAG_DIR"] = "/kaggle/working/oct_cds_outputs/oct_c8_densenet121/rag"
subprocess.Popen(
    ["streamlit", "run", "app/case_browser.py",
     "--server.port", "8501", "--server.headless", "true",
     "--browser.gatherUsageStats", "false"],
    stdout=open("/tmp/st.log", "w"), stderr=subprocess.STDOUT,
)
time.sleep(5)
tun = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://localhost:8501"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in tun.stdout:
    print(line, end="")
    if m := re.search(r"https://[-\w.]+trycloudflare\.com", line):
        print("\n\n>>> OPEN:", m.group(0))
        break
```

Open the printed `https://…trycloudflare.com` URL. Keep the cell running (the
tunnel dies when it stops). `localtunnel` or `ngrok` work the same way if you
prefer them.

## Notes

- The JSONL stores absolute `image_path`s (`/kaggle/input/datasets/.../bscans/*.png`).
  Those resolve inside the Kaggle session that produced them; elsewhere the app
  shows a "image not found" placeholder and everything else still works.
- `@st.cache_data` means re-selecting cases is instant; edit the sidebar path or
  press `R` to reload after a new `rag_narrate.py` run.
