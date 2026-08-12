/*
  script.js
  ---------------------------------------------------------------------------
  Controller for the FastAPI Documentation Assistant chat UI.
  Plain vanilla JavaScript - no frameworks, no build step.

  Talks to exactly two backend endpoints (unchanged from the original app):
    GET  /health  -> { status, vector_db_ready, llm_ready }
    POST /ask     -> { answer, sources: [{document_name, file_name, file_path, chunk_number}] }

  Sections:
    1. Element references
    2. Theme (dark/light) handling
    3. Health status polling
    4. Markdown-lite rendering for answers (escape-first, then format)
    5. Message rendering (user / assistant / typing / error)
    6. Composer behaviour (submit, autosize, keyboard shortcuts)
    7. Toast notifications
    8. Wiring / init
  ---------------------------------------------------------------------------
*/

(function () {
    "use strict";

    /* ---------------------------------------------------------------------
       1. Element references
       --------------------------------------------------------------------- */

    const chatLog = document.getElementById("chat-log");
    const emptyState = document.getElementById("empty-state");
    const suggestions = document.getElementById("suggestions");

    const composerForm = document.getElementById("composer-form");
    const questionInput = document.getElementById("question-input");
    const askButton = document.getElementById("ask-button");

    const statusPill = document.getElementById("status-pill");
    const statusText = document.getElementById("status-text");

    const themeToggle = document.getElementById("theme-toggle");
    const clearButton = document.getElementById("clear-btn");

    const toast = document.getElementById("toast");

    const THEME_KEY = "fda-theme";
    const HEALTH_POLL_INTERVAL_MS = 30000;

    let messageCounter = 0;
    let isSending = false;
    let typingIndicatorEl = null;

    /* ---------------------------------------------------------------------
       2. Theme handling
       --------------------------------------------------------------------- */

    function getTheme() {
        return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
    }

    function setTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        try {
            localStorage.setItem(THEME_KEY, theme);
        } catch (e) {
            /* localStorage unavailable (private mode, etc.) - theme just won't persist */
        }
    }

    function toggleTheme() {
        setTheme(getTheme() === "dark" ? "light" : "dark");
    }

    /* ---------------------------------------------------------------------
       3. Health status polling
       --------------------------------------------------------------------- */

    async function checkHealth() {
        try {
            const response = await fetch("/health");
            const data = await response.json();

            if (data.status === "ok") {
                setStatus("ready", "Ready");
            } else if (!data.vector_db_ready) {
                setStatus("down", "Docs not indexed");
            } else if (!data.llm_ready) {
                setStatus("down", "LLM not configured");
            } else {
                setStatus("down", "Not ready");
            }
        } catch (error) {
            setStatus("down", "Server unreachable");
        }
    }

    function setStatus(state, label) {
        statusPill.setAttribute("data-state", state);
        statusText.textContent = label;
    }

    /* ---------------------------------------------------------------------
       4. Markdown-lite rendering
       Escapes HTML first (never trust model output as markup), then applies
       a small set of safe formatting rules: fenced code blocks, inline code,
       bold text, and paragraph breaks.
       --------------------------------------------------------------------- */

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function renderFormattedAnswer(rawText) {
        const escaped = escapeHtml(rawText);
        const codeBlocks = [];

        // Pull out fenced code blocks first so their contents are never
        // touched by inline formatting rules below.
        let withoutBlocks = escaped.replace(/```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g, function (match, lang, code) {
            const token = `\u0000CODEBLOCK${codeBlocks.length}\u0000`;
            codeBlocks.push(code.replace(/\n$/, ""));
            return token;
        });

        // Inline `code`
        withoutBlocks = withoutBlocks.replace(/`([^`\n]+)`/g, "<code>$1</code>");

        // **bold**
        withoutBlocks = withoutBlocks.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");

        // Restore code blocks as <pre><code>
        withoutBlocks = withoutBlocks.replace(/\u0000CODEBLOCK(\d+)\u0000/g, function (match, index) {
            return `<pre><code>${codeBlocks[Number(index)]}</code></pre>`;
        });

        return withoutBlocks;
    }

    /* ---------------------------------------------------------------------
       5. Message rendering
       --------------------------------------------------------------------- */

    function assistantAvatarSvg() {
        return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" fill="currentColor"/>
        </svg>`;
    }

    function formatTime(date) {
        try {
            return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
        } catch (e) {
            return "";
        }
    }

    function hideEmptyState() {
        if (emptyState) {
            emptyState.style.display = "none";
        }
    }

    function scrollToBottom() {
        requestAnimationFrame(function () {
            const viewport = chatLog.parentElement;
            viewport.scrollTop = viewport.scrollHeight;
        });
    }

    function addUserMessage(text) {
        hideEmptyState();

        const el = document.createElement("div");
        el.className = "message user";
        el.innerHTML = `
            <div class="bubble-col">
                <div class="bubble"></div>
                <div class="msg-actions">
                    <span class="msg-time">${formatTime(new Date())}</span>
                </div>
            </div>
        `;
        el.querySelector(".bubble").textContent = text;

        chatLog.appendChild(el);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const el = document.createElement("div");
        el.className = "message assistant";
        el.id = "typing-indicator";
        el.innerHTML = `
            <div class="avatar">${assistantAvatarSvg()}</div>
            <div class="bubble-col">
                <div class="bubble typing-bubble">
                    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
                </div>
            </div>
        `;
        chatLog.appendChild(el);
        typingIndicatorEl = el;
        scrollToBottom();
    }

    function removeTypingIndicator() {
        if (typingIndicatorEl) {
            typingIndicatorEl.remove();
            typingIndicatorEl = null;
        }
    }

    function buildSourcesPanel(sources) {
        if (!sources || sources.length === 0) {
            return "";
        }

        const rows = sources.map(function (source, index) {
            const fileName = source.document_name || source.file_name || "unknown";
            const chunk = source.chunk_number !== undefined && source.chunk_number !== null ? source.chunk_number : "N/A";
            const filePath = source.file_path || "";
            return `
                <div class="source-row">
                    <span class="source-index">${index + 1}</span>
                    <span class="source-file">${escapeHtml(String(fileName))}</span>
                    <span class="source-chunk">chunk ${escapeHtml(String(chunk))}</span>
                    <span class="source-path">${escapeHtml(String(filePath))}</span>
                </div>
            `;
        }).join("");

        return `
            <div class="response-panel">
                <button type="button" class="response-panel-toggle">
                    <span class="response-status">200</span>
                    <span>Sources &middot; ${sources.length}</span>
                    <svg class="chevron" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="m6 9 6 6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </button>
                <div class="response-panel-body">${rows}</div>
            </div>
        `;
    }

    function addAssistantMessage(answer, sources, isError) {
        hideEmptyState();
        messageCounter += 1;
        const messageId = "msg-" + messageCounter;

        const el = document.createElement("div");
        el.className = "message assistant" + (isError ? " is-error" : "");
        el.innerHTML = `
            <div class="avatar">${assistantAvatarSvg()}</div>
            <div class="bubble-col">
                <div class="bubble">${renderFormattedAnswer(answer)}</div>
                ${isError ? "" : buildSourcesPanel(sources)}
                <div class="msg-actions">
                    <button type="button" class="msg-action copy-btn">Copy</button>
                    <span class="msg-time">${formatTime(new Date())}</span>
                </div>
            </div>
        `;

        const panelToggle = el.querySelector(".response-panel-toggle");
        if (panelToggle) {
            panelToggle.addEventListener("click", function () {
                panelToggle.closest(".response-panel").classList.toggle("is-open");
            });
        }

        const copyBtn = el.querySelector(".copy-btn");
        if (copyBtn) {
            copyBtn.addEventListener("click", function () {
                copyToClipboard(answer, copyBtn);
            });
        }

        el.id = messageId;
        chatLog.appendChild(el);
        scrollToBottom();
    }

    function copyToClipboard(text, buttonEl) {
        if (!navigator.clipboard) {
            return;
        }
        navigator.clipboard.writeText(text).then(function () {
            const original = buttonEl.textContent;
            buttonEl.textContent = "Copied";
            setTimeout(function () {
                buttonEl.textContent = original;
            }, 1400);
        }).catch(function () {
            showToast("Could not copy to clipboard.");
        });
    }

    /* ---------------------------------------------------------------------
       6. Composer behaviour
       --------------------------------------------------------------------- */

    function autoResizeInput() {
        questionInput.style.height = "auto";
        questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
    }

    async function handleSubmit(event) {
        event.preventDefault();

        const question = questionInput.value.trim();
        if (!question || isSending) {
            return;
        }

        addUserMessage(question);
        questionInput.value = "";
        autoResizeInput();

        setSending(true);
        showTypingIndicator();

        try {
            const response = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question }),
            });

            const data = await response.json();
            removeTypingIndicator();

            if (!response.ok) {
                addAssistantMessage(data.answer || "Something went wrong. Please try again.", [], true);
            } else {
                addAssistantMessage(data.answer, data.sources, false);
            }
        } catch (error) {
            removeTypingIndicator();
            addAssistantMessage("Could not reach the server. Check your connection and try again.", [], true);
            showToast("Network error — could not reach the server.");
        } finally {
            setSending(false);
        }
    }

    function setSending(sending) {
        isSending = sending;
        askButton.disabled = sending;
    }

    function handleComposerKeydown(event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            composerForm.requestSubmit();
        }
    }

    function askSuggestion(text) {
        questionInput.value = text;
        autoResizeInput();
        composerForm.requestSubmit();
    }

    function clearConversation() {
        chatLog.querySelectorAll(".message").forEach(function (el) {
            el.remove();
        });
        if (emptyState) {
            emptyState.style.display = "";
        }
        questionInput.focus();
    }

    /* ---------------------------------------------------------------------
       7. Toast notifications
       --------------------------------------------------------------------- */

    let toastTimer = null;

    function showToast(message) {
        toast.textContent = message;
        toast.classList.add("show");

        clearTimeout(toastTimer);
        toastTimer = setTimeout(function () {
            toast.classList.remove("show");
        }, 3200);
    }

    /* ---------------------------------------------------------------------
       8. Wiring / init
       --------------------------------------------------------------------- */

    themeToggle.addEventListener("click", toggleTheme);
    clearButton.addEventListener("click", clearConversation);

    composerForm.addEventListener("submit", handleSubmit);
    questionInput.addEventListener("keydown", handleComposerKeydown);
    questionInput.addEventListener("input", autoResizeInput);

    if (suggestions) {
        suggestions.addEventListener("click", function (event) {
            const button = event.target.closest(".suggestion");
            if (button) {
                askSuggestion(button.getAttribute("data-q"));
            }
        });
    }

    checkHealth();
    setInterval(checkHealth, HEALTH_POLL_INTERVAL_MS);
    questionInput.focus();
})();
