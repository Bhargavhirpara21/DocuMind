# Frontend Manual Test Cases

Use these prompts from the Streamlit UI to test answer quality, citations, and upload refresh behavior. This is separate from `evaluation/test_questions.json`.

## How To Run The App Right Now

1. Open PowerShell in `C:\Users\fk6147\DocuMind`.
2. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Start the API in one terminal:

```powershell
python -m uvicorn src.api.routes:app --reload --port 8000
```

4. Start the Streamlit frontend in a second terminal:

```powershell
streamlit run frontend/app.py
```

5. Open `http://localhost:8501` in your browser.

If you want to change the API host, set `DOCUMIND_API_URL` before starting Streamlit. By default it uses `http://localhost:8000`.

## Ask Questions Page

| ID | Prompt | What to verify |
| --- | --- | --- |
| F-01 | What does the cover say the application area is? | Returns Aerospace and cites page 1 of `flyer-small-part-aero-2026-en.pdf`. |
| F-02 | What contact website is shown on page 2? | Returns `walter-tools.com` and cites page 2. |
| F-03 | What is the first entry in the table of contents? | Returns `A1 - ISO turning` and cites page 3. |
| F-04 | Which manufacturing categories are listed on the cover? | Returns `Turning, Holemaking, Threading` for the aero flyer. |
| F-05 | What title is shown on the cover of the highlights brochure? | Returns `Product highlights` and cites page 1 of `product-innovations-2026-1-highlight-en.pdf`. |
| F-06 | What phrase appears on the cover of the highlights brochure? | Returns `A passion to win.` and cites page 1. |
| F-07 | Which product grades are called out first in the ISO turning section? | Returns `Tigertec Gold turning grades WMP20G, WMP30G` and cites page 3. |
| F-08 | Which Groovtec grooving system is listed in the highlights brochure? | Returns `Groov'tec GD grooving system G5011` and cites page 3. |
| F-09 | What section starts on page 4 of the product innovations catalog? | Returns `D1 - VHM-, PKD- und Keramik-Fräswerkzeuge` and cites page 4. |
| F-10 | Do the documents mention a warranty period for WMP20G? | The assistant should say it cannot find that information instead of guessing. |
| F-11 | Give me a one-sentence summary of the cover and cite the source. | The answer should be brief, grounded, and include a citation. |
| F-12 | Which document says `A passion to win.`? | Returns the highlights brochure and cites the right source. |

## Upload & Ask Page

| ID | Action | Prompt | What to verify |
| --- | --- | --- | --- |
| U-01 | Upload a PDF, then ask | What title is shown on the cover? | The uploaded file should be ingested and the answer should refresh after upload. |
| U-02 | Upload a PDF, then ask | What website is listed on page 2? | The response should come from the uploaded document, not the old index. |
| U-03 | Upload a PDF, then ask a follow-up question | What is the first section listed in the table of contents? | The sidebar should update and the chat history should keep the conversation readable. |

## Suggested Order

If you want a fast smoke test, run `F-01`, `F-02`, `F-05`, `F-07`, `F-08`, and `F-10` first. Those cover the exact retrieval paths that were weakest before.