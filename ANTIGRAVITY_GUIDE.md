# How to Use BrevardVectorDB with Google Antigravity
### A Step-by-Step Guide for Complete Beginners (Zero AI or Coding Experience Needed)

---

## Welcome! What Does This Do?

Imagine having a personal research assistant who has:
1. **Watched all 80 Brevard Public Schools Board Meetings and Work Sessions** (from 2024 through late 2026) and written down every word spoken, who said it, and the exact second on the video it happened.
2. **Read all 1,663 official meeting agendas, minutes, and attached PDF contracts/presentations.**
3. **Memorized 2,422 sections of Florida Law** covering public education, teachers' unions, public employment, public records, and criminal statutes.

With **Google Antigravity** and **BrevardVectorDB**, you can ask questions in plain English—just like typing a text message—and get instant, factual answers with **clickable YouTube links that jump to the exact second in the meeting**, official PDF citations, and Florida Statute references.

---

## ⚡ The 30-Second Shortcut: Just Copy & Paste Into Google Antigravity!

If you already have **[Google Antigravity](https://antigravity.google)** open on your computer, you don't even have to download files manually. **Just copy the box below (or copy and paste this entire guide!) directly into the Google Antigravity chat box and press Enter:**

```text
Hello Antigravity! Please set up and connect the BrevardVectorDB dataset for me so I can ask questions in plain English.

Follow these automatic setup steps on my computer:
1. Check if `unpack_db.py` and `db_archive/` already exist in the current workspace folder.
   - If NOT, download and extract `https://github.com/Roach758/BrevardVectorDB/archive/refs/heads/main.zip` (or run `git clone https://github.com/Roach758/BrevardVectorDB.git`) into my current workspace folder.
2. Make sure Python and the required libraries (`chromadb`, `sentence-transformers`) are installed (`pip install -r requirements.txt`).
3. Run `python unpack_db.py` to reassemble the compressed database parts in `db_archive/` into `chroma_storage/` (54,054 vectors), `videos.db` (80 school board videos), and `laws.db` (2,422 Florida Statutes).
4. Run `python search_db.py --stats` to verify all collections are online.
5. Whenever I ask a question about Brevard Public Schools, board meetings, agendas/PDFs, or Florida laws, run `python search_db.py "<query>"` (or `--keyword` if offline/CPU-only) and cite the exact Meeting Title, Date, Speaker, clickable YouTube timestamp link (`https://www.youtube.com/watch?v=<video_id>&t=<start_time>s`), agenda PDF name, and Florida Statute section.

When setup finishes, show me the database stats and ask me what topic I would like to investigate first!
```

---

## How It Works (In Plain English)

```
+-------------------------------------------------------------------+
|  1. YOU type a question in plain English inside Google Antigravity |
|     Example: "What did the Board decide about cell phones?"       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|  2. GOOGLE ANTIGRAVITY searches BrevardVectorDB on your computer  |
|     - 5,454 Video Transcript chunks (80 full meetings)            |
|     - 39,165 Official Agenda & PDF chunks (1,663 documents)       |
|     - 9,435 Florida Statute chunks (69 chapters / 2,422 laws)     |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|  3. YOU receive a clear summary with verifiable proof:            |
|     - Direct YouTube links stamped to the exact minute & second   |
|     - Official Agenda / PDF document names and dates              |
|     - Exact Florida Statute numbers (e.g., Fla. Stat. s. 1006.07) |
+-------------------------------------------------------------------+
```

---

## Step-by-Step Setup (Takes About 5 Minutes)

### Step 1: Install Google Antigravity
1. Open your web browser and go to **[https://antigravity.google](https://antigravity.google)**.
2. Click the **Download** button and install Google Antigravity on your computer.
3. Open Google Antigravity and sign in with your Google account.

### Step 2: Install Python (The Database Reader)
Google Antigravity uses a free program called **Python** behind the scenes to open the database files. *(You will never have to write Python code yourself!)*
1. Go to **[https://www.python.org/downloads/](https://www.python.org/downloads/)**.
2. Click the yellow **Download Python** button and open the installer.
3. **CRITICAL STEP FOR WINDOWS USERS:** On the very first installer screen, check the box at the bottom that says:
   > **`☑ Add python.exe to PATH`**
4. Click **Install Now** and wait for it to finish.

### Step 3: Download the BrevardVectorDB Folder
1. Go to **[https://github.com/Roach758/BrevardVectorDB](https://github.com/Roach758/BrevardVectorDB)**.
2. Click the green **`<> Code`** button near the top right of the page, then click **`Download ZIP`**.
3. Open your computer's **Downloads** folder:
   - **Windows:** Right-click `BrevardVectorDB-main.zip`, choose **Extract All...**, and click **Extract**.
   - **Mac:** Double-click `BrevardVectorDB-main.zip`.
4. Move the extracted folder to your **Desktop** or **Documents** folder so it is easy to find.

### Step 4: Open the Folder in Google Antigravity
1. Open **Google Antigravity**.
2. Open your downloaded folder:
   - If you are using the **Antigravity Desktop App**: Click **Projects** on the left sidebar -> click **Open Project** -> choose your **`BrevardVectorDB`** folder.
   - If you are using **Antigravity IDE**: Click **File** in the top-left menu -> click **Open Folder...** -> choose your **`BrevardVectorDB`** folder.

### Step 5: One-Sentence Automatic Setup
The first time you open the folder, tell Antigravity to unpack the database for you:
1. Click inside the **Chat Box** in Google Antigravity.
2. Copy and paste this exact sentence and press **Enter**:
   ```text
   Please install the Python packages and unpack the database for me, then show me the database stats.
   ```
3. If a button pops up asking for permission to run a command (such as `Proceed`, `Allow`, or `Run`), click **Allow / Proceed**.
4. Antigravity will automatically reassemble the database files and display a confirmation table showing all **54,054 vectors**, **80 videos**, and **2,422 Florida laws**.

---

## How to Ask Great Questions

You don't need special keywords or computer commands. Just talk to Google Antigravity naturally! Here are some tips to get the best results:

### 1. Ask for Timestamps and YouTube Links
> *"Find every time board members discussed book challenges or library materials. Give me a summary of each meeting along with the exact YouTube link and timestamp."*

### 2. Compare What Was Said vs. Written Agendas
> *"Compare what was written in the agenda attachments about the budget or millage rate with what the board members actually said during the video meeting."*

### 3. Fact-Check Decisions Against Florida Law
> *"Look up Florida Statutes on student expulsion and code of student conduct in Chapter 1006, and check how those statutes were discussed during Brevard School Board work sessions."*

### 4. Ask for Tables or Timelines
> *"Create a timeline table of every discussion about teacher pay and union bargaining across 2025 and 2026 meetings, including the meeting date, key speakers, and YouTube link."*

---

## Understanding the Google Antigravity Screen

When you use Google Antigravity, you will see a few helpful areas on your screen:

* **The Left Sidebar:** Lets you start a **New Conversation** whenever you want to switch to a fresh topic.
* **The Chat Canvas (Center):** Where you type your questions and read Antigravity's answers.
* **The Artifacts & Auxiliary Pane (Right):** When Antigravity writes a long research report, table, or investigation summary, it can display it here as a clean, printable document called an **Artifact**.
* **Slash Commands (Type `/` in the chat box):**
  * `/plan` — Use this when you have a complex question and want Antigravity to show you its research plan before starting.
  * `/goal` — Use this when you want Antigravity to do a deep, exhaustive investigation across dozens of meetings without stopping early.
  * `/compact` — Use this if your conversation gets very long and you want Antigravity to tidy up the chat history.

---

## Frequently Asked Questions (FAQ)

### Q: Does this send my database to the cloud, or is the database on my computer?
**A:** The entire database (**80 video transcripts**, **1,663 meeting documents**, and **2,422 Florida Statutes**) lives right on your own hard drive inside the `BrevardVectorDB` folder.

### Q: Do I need an expensive graphics card (GPU) to search the database?
**A:** **No!** The heavy work of transcribing 80 videos and vectorizing 54,054 text passages has **already been done for you** and saved inside the database. Any normal laptop or desktop computer can search the pre-built database in seconds. Our search tool (`search_db.py`) even includes an instant keyword mode (`--keyword`) that runs in milliseconds on any computer.

### Q: What if Antigravity asks me to click "Allow" or "Proceed"?
**A:** Google Antigravity has a built-in safety feature called a sandbox that asks your permission before running a search script on your computer. When it runs `python search_db.py` or `python unpack_db.py`, clicking **Allow** lets it read the database to answer your question.

### Q: How can I verify that the AI isn't making things up?
**A:** Every video transcript result includes a clickable YouTube link formatted like `https://www.youtube.com/watch?v=K5GTUavrs7w&t=1420s`. Clicking that link opens the official Brevard Public Schools YouTube recording at the **exact second** the quote was spoken so you can watch and verify it with your own eyes and ears.
