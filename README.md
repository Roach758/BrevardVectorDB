# BrevardVectorDB

**A Complete Vector & Relational Database of Brevard Public Schools (BPS) Board Meetings, Official Agendas, Attachments, and Florida Statutes — Pre-Configured for Google Antigravity & Grokbot**

[![Vectors](https://img.shields.io/badge/ChromaDB_Vectors-54%2C054-0A66C2?style=for-the-badge)](#dataset-at-a-glance)
[![Meetings](https://img.shields.io/badge/Video_Meetings-80_Ingested-2EA44F?style=for-the-badge)](#dataset-at-a-glance)
[![Documents](https://img.shields.io/badge/Agendas_%26_PDFs-1%2C663_Files-D97706?style=for-the-badge)](#dataset-at-a-glance)
[![Statutes](https://img.shields.io/badge/Florida_Statutes-2%2C422_Sections-7C3AED?style=for-the-badge)](#dataset-at-a-glance)
[![Configured For](https://img.shields.io/badge/Ready_For-Google_Antigravity_%26_Grokbot-E11D48?style=for-the-badge)](#complete-beginners-guide-using-this-dataset-with-google-antigravity)

---

## ⚡ Instant Start: Copy & Paste Into Google Antigravity

> **Never used AI or GitHub before?**
> You do **not** need to download or configure anything manually. Just install **[Google Antigravity](https://antigravity.google)**, open a chat window, and **copy & paste the box below (or copy and paste this entire README page!) directly into the Google Antigravity chat box:**

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

*(Note: When Antigravity asks permission to run these setup commands on your computer, simply click **Allow** or **Proceed**.)*

---

## Table of Contents

1. [What Is BrevardVectorDB?](#what-is-brevardvectordb)
2. [Dataset at a Glance](#dataset-at-a-glance)
3. [Complete Beginner's Guide: Using This Dataset with Google Antigravity (No AI Experience Required)](#complete-beginners-guide-using-this-dataset-with-google-antigravity)
   - [Step 1: Install Google Antigravity & Python](#step-1-install-google-antigravity--python)
   - [Step 2: Download This Dataset to Your Computer](#step-2-download-this-dataset-to-your-computer)
   - [Step 3: Open the Folder in Google Antigravity](#step-3-open-the-folder-in-google-antigravity)
   - [Step 4: Tell Antigravity to Prepare the Database (One Sentence!)](#step-4-tell-antigravity-to-prepare-the-database-one-sentence)
   - [Step 5: Ask Questions in Plain English](#step-5-ask-questions-in-plain-english)
   - [Copy-and-Paste Example Prompts](#copy-and-paste-example-prompts)
   - [Helpful Google Antigravity Features for Beginners](#helpful-google-antigravity-features-for-beginners)
4. [Command-Line & Python Quickstart (For Developers & Grokbot)](#command-line--python-quickstart-for-developers--grokbot)
5. [Database Architecture & Schema Reference](#database-architecture--schema-reference)
6. [Adding New Meetings or Re-Vectorizing Data](#adding-new-meetings-or-re-vectorizing-data)

---

## What Is BrevardVectorDB?

**BrevardVectorDB** is an open, local civic-intelligence dataset built via the **Project Cicero** pipeline. It turns hundreds of hours of school board video recordings, thousands of pages of board agendas and PDF attachments, and the full body of relevant Florida Statutes into an immediately searchable AI knowledge base.

Instead of manually scrubbing through 3-hour YouTube videos or reading hundreds of PDF attachments, you can ask plain-English questions in **Google Antigravity** (or via **Grokbot** / Python) and receive answers backed by:
- **Exact YouTube video links stamped to the second** a statement was spoken (`https://www.youtube.com/watch?v=...&t=...s`)
- **Official written agenda and attachment citations** (file names, dates, and page numbers)
- **Exact Florida Statute citations** (`Fla. Stat. s. 1006.07`, etc.) to cross-check board policy against state law

---

## Dataset at a Glance

All vectors are embedded with **`BAAI/bge-base-en-v1.5`** (768-dimensional normalized cosine embeddings) and persisted locally in **ChromaDB** (`chroma_storage/`) alongside two **SQLite** relational databases (`videos.db` and `laws.db`).

| Data Source | Storage Target | Collection / Table | Volume | What It Contains |
| :--- | :--- | :--- | :--- | :--- |
| **School Board Video Transcripts (2024–2026)** | ChromaDB + `videos.db` | `transcripts` / `videos` | **80 Videos**<br>**5,454 Vector Chunks** | Full speech-to-text transcripts (`faster-whisper`) with speaker diarization (`WavLM`), timestamps, and direct YouTube links |
| **Board Agendas, Minutes & PDF Attachments** | ChromaDB | `meeting_documents` | **1,663 Files**<br>**39,165 Vector Chunks** | Every HTML agenda, HTML minutes document, and attached PDF packet from 2024 onward |
| **Florida Statutes (Education, Unions, Employment, Criminal)** | ChromaDB + `laws.db` | `florida_laws` / `chapters`, `sections` | **69 Chapters**<br>**2,422 Sections**<br>**9,435 Vector Chunks** | Full statutory text and legislative history for Chapters 119, 120, 447, 775–893, and 1000–1014 |
| **Combined Totals** | **Local Disk** | **3 Vector Collections + 2 SQLite DBs** | **54,054 Vector Chunks** | **100% self-contained local database** |

---

## Complete Beginner's Guide: Using This Dataset with Google Antigravity

> **Never used AI before? You are in the right place.**
> You do **not** need to know how to write computer code. If you can download a folder and type a question into a chat box—just like sending a text message or searching on Google—you can use this dataset.

### What Is Google Antigravity?

**Google Antigravity** is a free AI application from Google that runs on your computer. Unlike standard web chatbots that only know general internet trivia, Antigravity can **open a folder on your computer** and read the databases inside it.

We have pre-configured this folder with special built-in instructions (`.agents/rules/brevard-vector-db.md`) so that the moment you open it in Google Antigravity, the AI automatically knows how to search all **80 school board videos**, **1,663 meeting documents**, and **2,422 Florida laws** for you.

---

### Step 1: Install Google Antigravity & Python

Before downloading the dataset, make sure two free programs are installed on your computer:

1. **Google Antigravity**
   - Visit **[https://antigravity.google](https://antigravity.google)**
   - Click **Download** for your operating system (Windows, Mac, or Linux) and run the installer.
   - Open Google Antigravity and sign in with your Google account.

2. **Python (Version 3.11 or 3.12)** *(Antigravity uses Python behind the scenes to open the database files)*
   - Visit **[https://www.python.org/downloads/](https://www.python.org/downloads/)** and click the yellow **Download Python** button.
   - **Important for Windows users:** On the very first screen of the Python installer, check the box at the bottom that says **`☑ Add python.exe to PATH`**, then click **Install Now**.

---

### Step 2: Download This Dataset to Your Computer

1. On this GitHub page (`https://github.com/Roach758/BrevardVectorDB`), look toward the top-right corner and click the green **`<> Code`** button.
2. Click **`Download ZIP`** from the dropdown menu.
3. Once the `.zip` file finishes downloading to your **Downloads** folder:
   - **On Windows:** Right-click `BrevardVectorDB-main.zip` -> click **Extract All...** -> click **Extract**.
   - **On Mac:** Double-click `BrevardVectorDB-main.zip` to unzip it.
4. Move the extracted **`BrevardVectorDB`** folder somewhere easy to find, such as your **Desktop** or **Documents** folder.

*(Note for developers: You can also clone the repository directly via `git clone https://github.com/Roach758/BrevardVectorDB.git`.)*

---

### Step 3: Open the Folder in Google Antigravity

1. Launch **Google Antigravity** on your computer.
2. Open the **`BrevardVectorDB`** folder as your project workspace:
   - **In Antigravity 2.0 (Desktop App):** Click **Projects** in the left-hand sidebar -> click **Open Project / Folder** -> select your **`BrevardVectorDB`** folder.
   - **In Antigravity IDE:** Click **File** in the top menu bar -> click **Open Folder...** -> select your **`BrevardVectorDB`** folder.

---

### Step 4: Tell Antigravity to Prepare the Database (One Sentence!)

To fit smoothly on GitHub, the 795 MB database is stored inside the `db_archive/` folder in compressed pieces. You only need to unpack it **once**.

You don't even have to unpack it yourself—just ask Antigravity to do it!

1. Find the **Chat Box** in Google Antigravity.
2. Type the following sentence and press **Enter**:

   ```text
   Please set up this project, install requirements, and unpack the database for me.
   ```

3. **What happens next:**
   - Antigravity will automatically install the required database reader (`chromadb` and `sentence-transformers`) and run `python unpack_db.py`.
   - If Antigravity asks for your permission to run a terminal command, click **Allow** or **Proceed**.
   - Within about 30–60 seconds, Antigravity will reply that all **54,054 vector chunks**, **80 videos**, and **2,422 Florida Statutes** are unpacked and ready!

---

### Step 5: Ask Questions in Plain English

That's it! From now on, whenever you open this project in Google Antigravity, you can ask questions in plain English.

Behind the scenes, Antigravity will search the database (`search_db.py`), read the matching video transcripts, board documents, and Florida Statutes, and write a clear report for you with exact timestamps and clickable links.

---

### Copy-and-Paste Example Prompts

Try copying and pasting any of these questions directly into the Google Antigravity chat box:

#### Searching What Was Said in School Board Video Meetings
* *"Search the video transcripts for discussions about student cell phone policies. What were the main arguments, and can you give me the exact YouTube links and timestamps?"*
* *"What did speakers talk about during public comment at the November 17, 2026 School Board meeting?"*
* *"Find every meeting where school bus driver shortages or transportation delays were discussed. Summarize what was said in chronological order."*

#### Searching Official Written Agendas, Contracts & PDF Attachments
* *"Search the meeting documents and PDF attachments for procurement contracts or vendor renewals over $500,000 in 2025 and 2026."*
* *"What agenda items and attachments were included in the June 2025 School Board Work Sessions?"*

#### Cross-Checking Board Actions Against Florida Law
* *"What does Florida law say about student discipline and wireless communication devices in Chapter 1006, and how did the Brevard School Board discuss implementing those rules in their meetings?"*
* *"Explain the collective bargaining rules for public school employees under Florida Statute Chapter 447, and find any board meetings where union negotiations or salary schedules were discussed."*
* *"What are the public records and open meeting requirements in Florida Statutes Chapter 119 and 1001, and have those statutes ever been cited during a Brevard School Board meeting?"*

---

### Helpful Google Antigravity Features for Beginners

As you get comfortable chatting with your data, here are four built-in Google Antigravity features that make research even easier:

| Feature | How to Use It | Why It Is Useful |
| :--- | :--- | :--- |
| **`@` Mentions** | Type `@` in the chat box (e.g., `@videos.db` or `@search_db.py`) | Points the AI directly to a specific file or database table you want it to inspect. |
| **`/plan` Command** | Type `/plan` before a big research question | Tells Antigravity to outline a step-by-step research plan for your approval before it starts searching. |
| **`/goal` Command** | Type `/goal` before a deep investigation | Tells Antigravity to keep researching thoroughly across multiple meetings and laws until the entire investigation is complete. |
| **Artifacts Pane** | Ask *"Please save this report as an Artifact"* | Creates a clean, formatted document with tables, diagrams, and clickable YouTube links that you can read or export on the side of your screen. |

> **Want the detailed standalone guide?** See **[ANTIGRAVITY_GUIDE.md](ANTIGRAVITY_GUIDE.md)** for a printable walkthrough with troubleshooting tips.

---

## Command-Line & Python Quickstart (For Developers & Grokbot)

If you are a developer or integrating this dataset with **Grokbot**, **LangChain**, or custom scripts, you can use the included utilities directly from your terminal.

### 1. Install Dependencies & Unpack

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python unpack_db.py
```

### 2. Run Unified Searches via `search_db.py`

```bash
# View database counts across all collections and SQLite tables
python search_db.py --stats

# Semantic AI search across ALL 3 collections (Video Transcripts + Agendas/PDFs + Florida Laws)
python search_db.py "student cell phone discipline policy"

# Search ONLY video transcripts (returns YouTube links stamped to exact seconds)
python search_db.py "charter school funding" --collection transcripts --top-k 8

# Search ONLY official agendas, minutes & PDF attachments
python search_db.py "millage rate budget hearing" --collection meetings --top-k 8

# Search ONLY Florida Statutes (Education, Unions, Employment, Criminal Law)
python search_db.py "student expulsion procedures" --collection laws --top-k 5

# Instant exact keyword/phrase search (no GPU or embedding model required!)
python search_db.py "1006.07" --collection laws --keyword
```

### 3. Launch the Interactive Project Cicero Web App & FastAPI Server

```bash
# Start the FastAPI backend (port 8000)
uvicorn main:app --host 127.0.0.1 --port 8000

# In a second terminal, launch the Streamlit frontend
streamlit run project_cicero.py
```

---

## Database Architecture & Schema Reference

### 1. ChromaDB Persistent Vector Store (`chroma_storage/`)
- **Embedding Model:** `BAAI/bge-base-en-v1.5` (`cosine` distance space, normalized 768-dim vectors)
- **Query Prefix for BGE:** `"Represent this sentence for searching relevant passages: "`
- **Collections:**
  1. **`transcripts`** (`5,454` chunks)
     - **Metadata fields:** `video_id`, `title`, `channel`, `upload_date`, `duration`, `chunk_index`, `start_time` (seconds), `end_time` (seconds), `primary_speaker`, `speakers`
  2. **`meeting_documents`** (`39,165` chunks)
     - **Metadata fields:** `file_name`, `file_path`, `file_type` (`pdf` / `html`), `folder_name`, `meeting_date` (`YYYY-MM-DD`), `meeting_title`, `chunk_index`, `total_chunks`
  3. **`florida_laws`** (`9,435` chunks)
     - **Metadata fields:** `section` (e.g., `"1006.07"`), `chapter` (integer), `category`, `catchline`, `file_path`, `chunk_index`, `total_chunks`

### 2. SQLite Relational Databases
- **`videos.db`** — Table `videos`:
  - `video_id` (`TEXT PRIMARY KEY`), `title`, `url`, `channel`, `upload_date`, `duration` (seconds), `processed_date`
- **`laws.db`** — Tables `chapters` & `sections`:
  - `chapters`: `chapter` (`INTEGER PRIMARY KEY`), `category`, `title`, `file_path`, `sections_count`, `chunks_count`, `processed_date`
  - `sections`: `section` (`TEXT PRIMARY KEY`), `chapter`, `category`, `catchline`, `body` (full unchunked statutory text), `history`, `file_path`

---

## Adding New Meetings or Re-Vectorizing Data

- **Ingest a new YouTube School Board Meeting:**
  Add the YouTube URL to `Videos_Starting_in_2024.txt` and run:
  ```bash
  python batch_ingest.py
  ```
  *(Automatically skips already-ingested videos in `videos.db` and runs `yt-dlp` -> `faster-whisper` -> `WavLM` speaker diarization -> `BAAI/bge-base-en-v1.5` -> ChromaDB).*
- **Vectorize new Agenda HTMLs / PDFs:**
  Place new meeting folders in `agendas/` and run:
  ```bash
  python vectorize_meetings.py
  ```
- **Vectorize additional Florida Statutes:**
  Place statute text files in `Laws/` and run:
  ```bash
  python vectorize_laws.py
  ```
