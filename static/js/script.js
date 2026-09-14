const STORAGE_KEY = "faq_chatbot_history";
const MAX_HISTORY_MESSAGES = 100;
const REQUEST_TIMEOUT_MS = 15000;
const WELCOME_MESSAGE =
    "Hi! I'm your FAQ Assistant.\nI can answer questions about courses, enrollment, " +
    "payments, refunds, certificates, assignments, exams, and account issues. " +
    "Try one of the examples below or ask your own question.";
const EXAMPLE_QUESTIONS = [
    "How can I reset my password?",
    "How do I enroll in a course?",
    "How can I download my certificate?",
    "What is your refund policy?"
];

const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const statusIndicator = document.getElementById("status-indicator");
const statusText = document.getElementById("status-text");
const newChatButton = document.getElementById("new-chat-button");
const newChatDialog = document.getElementById("new-chat-dialog");
const dialogCancel = document.getElementById("dialog-cancel");
const dialogConfirm = document.getElementById("dialog-confirm");

let conversation = [];
let isRequesting = false;

document.addEventListener("DOMContentLoaded", initializeChat);

function initializeChat() {
    restoreConversation();
    checkApiHealth();
    chatForm.addEventListener("submit", handleSubmit);
    newChatButton.addEventListener("click", () => {
        if (conversation.length === 0) {
            startNewChat();
        } else {
            newChatDialog.showModal();
        }
    });
    newChatDialog.addEventListener("click", handleDialogClick);
    newChatDialog.addEventListener("cancel", (event) => {
        event.preventDefault();
        newChatDialog.close();
    });
}

function restoreConversation() {
    const stored = loadConversation();
    if (stored.length > 0) {
        conversation = trimHistory(stored);
        renderConversation();
    } else {
        conversation = [];
        renderMessage(createWelcomeMessage());
        renderSuggestionChips();
    }
    scrollToLatestMessage();
}

async function checkApiHealth() {
    setStatus("offline", "Offline");
    try {
        const apiResponse = await fetch("/api/health");
        if (!apiResponse.ok) {
            setStatus("offline", "Offline");
            return;
        }
        const data = await apiResponse.json();
        setStatus(
            data.status === "ok" ? "online" : "offline",
            data.status === "ok" ? "Online" : "Offline"
        );
    } catch (error) {
        setStatus("offline", "Offline");
    }
}

function setStatus(state, label) {
    statusIndicator.classList.toggle("is-online", state === "online");
    statusIndicator.classList.toggle("is-offline", state === "offline");
    statusText.textContent = label;
}

/* ---------- Empty state ---------- */

function renderSuggestionChips() {
    const container = document.createElement("div");
    container.className = "suggestion-chips";
    container.setAttribute("role", "group");
    container.setAttribute("aria-label", "Example questions");
    EXAMPLE_QUESTIONS.forEach((question) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "suggestion-chip";
        chip.textContent = question;
        chip.addEventListener("click", () => {
            userInput.value = question;
            chatForm.requestSubmit();
        });
        container.appendChild(chip);
    });
    chatMessages.appendChild(container);
    scrollToLatestMessage();
}

function removeSuggestionChips() {
    const container = chatMessages.querySelector(".suggestion-chips");
    if (container) {
        container.remove();
    }
}

async function handleSubmit(event) {
    event.preventDefault();

    if (isRequesting) {
        return;
    }

    const question = userInput.value.trim();
    if (!question) {
        userInput.focus();
        return;
    }

    userInput.value = "";
    addMessage(createMessage("user", question));
    isRequesting = true;
    setLoadingState(true);
    showTypingIndicator();

    try {
        const data = await sendMessage(question);
        if (data && data.error) {
            handleApiError(data.error, null);
        } else {
            hideTypingIndicator();
            addMessage(
                createMessage("bot", data.answer, {
                    matched: data.matched,
                    score: data.score,
                    category: data.category
                })
            );
        }
    } catch (error) {
        handleApiError(getErrorMessage(error), question);
    } finally {
        hideTypingIndicator();
        setLoadingState(false);
        isRequesting = false;
        userInput.focus();
    }
}

async function sendMessage(question) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    let response;
    try {
        response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: question }),
            signal: controller.signal
        });
    } finally {
        clearTimeout(timer);
    }

    let data = null;
    try {
        data = await response.json();
    } catch (error) {
        data = null;
    }

    if (!response.ok) {
        if (data && typeof data.error === "string") {
            return data;
        }
        throw new Error("chatbot request failed");
    }

    if (!isValidBotResponse(data)) {
        throw createMalformedResponseError();
    }

    return data;
}

function isValidBotResponse(data) {
    return (
        data !== null &&
        typeof data === "object" &&
        typeof data.matched === "boolean" &&
        typeof data.answer === "string" &&
        typeof data.score === "number" &&
        (data.faq_id === null || typeof data.faq_id === "number") &&
        (data.category === null || typeof data.category === "string")
    );
}

function createMalformedResponseError() {
    const error = new Error("malformed response");
    error.code = "malformed-response";
    return error;
}

function getErrorMessage(error) {
    if (error && error.name === "AbortError") {
        return "Request timed out. Please try again.";
    }
    if (error && error.code === "malformed-response") {
        return "Something went wrong while reading the chatbot response.";
    }
    return "Unable to connect to the chatbot. Please try again.";
}

/* ---------- Conversation state ---------- */

function createMessage(role, content, extra) {
    const message = {
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 8),
        role: role,
        content: content,
        timestamp: new Date().toISOString()
    };
    if (extra) {
        Object.assign(message, extra);
    }
    return message;
}

function createWelcomeMessage() {
    return {
        id: "welcome",
        role: "bot",
        content: WELCOME_MESSAGE,
        timestamp: new Date().toISOString()
    };
}

function addMessage(message) {
    conversation.push(message);
    conversation = trimHistory(conversation);
    saveConversation();
    renderMessage(message);
    if (message.role === "user") {
        removeSuggestionChips();
    }
    if (message.id !== "welcome") {
        scrollToLatestMessage();
    }
}

function renderConversation() {
    chatMessages.innerHTML = "";
    conversation.forEach(renderMessage);
}

function renderMessage(message) {
    const element = document.createElement("div");
    element.className = "message " + message.role + "-message";

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.content;

    const timeText = formatMessageTime(message.timestamp);
    if (timeText) {
        const time = document.createElement("span");
        time.className = "message-time";
        time.textContent = timeText;
        content.appendChild(time);
    }

    element.appendChild(content);
    chatMessages.appendChild(element);
}

function trimHistory(messages) {
    if (messages.length <= MAX_HISTORY_MESSAGES) {
        return messages;
    }
    return messages.slice(-MAX_HISTORY_MESSAGES);
}

/* ---------- Storage ---------- */

function saveConversation() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(conversation));
    } catch (error) {
        // Storage unavailable or full: the conversation stays usable in memory.
    }
}

function loadConversation() {
    let stored = null;
    try {
        stored = localStorage.getItem(STORAGE_KEY);
    } catch (error) {
        return [];
    }

    if (!stored) {
        return [];
    }

    try {
        const parsed = JSON.parse(stored);
        if (!Array.isArray(parsed)) {
            return [];
        }
        return parsed.filter(isValidMessage);
    } catch (error) {
        return [];
    }
}

function isValidMessage(message) {
    return (
        message !== null &&
        typeof message === "object" &&
        (message.role === "user" || message.role === "bot") &&
        typeof message.content === "string"
    );
}

function clearStoredConversation() {
    try {
        localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
        // Ignore: nothing else to do if storage is unavailable.
    }
}

/* ---------- New Chat ---------- */

function handleDialogClick(event) {
    if (event.target === dialogCancel) {
        newChatDialog.close();
    } else if (event.target === dialogConfirm) {
        startNewChat();
    }
}

function startNewChat() {
    conversation = [];
    clearStoredConversation();
    chatMessages.innerHTML = "";
    renderMessage(createWelcomeMessage());
    renderSuggestionChips();
    newChatDialog.close();
    userInput.focus();
}

/* ---------- Errors and retry ---------- */

function handleApiError(message, retryQuestion) {
    const element = document.createElement("div");
    element.className = "message error-message";

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message;

    if (retryQuestion) {
        element.dataset.question = retryQuestion;
        const retryButton = document.createElement("button");
        retryButton.className = "retry-button";
        retryButton.type = "button";
        retryButton.textContent = "Retry";
        retryButton.addEventListener("click", () => retryMessage(element, retryQuestion));
        content.appendChild(retryButton);
    }

    element.appendChild(content);
    chatMessages.appendChild(element);
    scrollToLatestMessage();
}

async function retryMessage(errorElement, question) {
    if (isRequesting) {
        return;
    }

    errorElement.remove();
    isRequesting = true;
    setLoadingState(true);
    showTypingIndicator();

    try {
        const data = await sendMessage(question);
        if (data && data.error) {
            hideTypingIndicator();
            handleApiError(data.error, null);
        } else {
            hideTypingIndicator();
            addMessage(
                createMessage("bot", data.answer, {
                    matched: data.matched,
                    score: data.score,
                    category: data.category
                })
            );
        }
    } catch (error) {
        hideTypingIndicator();
        handleApiError(getErrorMessage(error), question);
    } finally {
        setLoadingState(false);
        isRequesting = false;
        userInput.focus();
    }
}

/* ---------- UI helpers ---------- */

function formatMessageTime(timestamp) {
    try {
        const date = new Date(timestamp);
        if (Number.isNaN(date.getTime())) {
            return null;
        }
        return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    } catch (error) {
        return null;
    }
}

function showTypingIndicator() {
    const message = document.createElement("div");
    message.className = "message bot-message typing-indicator";
    message.setAttribute("aria-hidden", "true");

    const content = document.createElement("div");
    content.className = "message-content";

    const dots = document.createElement("span");
    dots.className = "typing-dots";

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        dot.className = "dot";
        dots.appendChild(dot);
    }

    content.appendChild(dots);
    message.appendChild(content);
    chatMessages.appendChild(message);
    scrollToLatestMessage();
    announceLoading(true);
}

function hideTypingIndicator() {
    const indicator = chatMessages.querySelector(".typing-indicator");
    if (indicator) {
        indicator.remove();
    }
    announceLoading(false);
}

function announceLoading(isLoading) {
    const region = document.getElementById("sr-loading");
    if (region) {
        region.textContent = isLoading
            ? "Processing your question…"
            : "";
    }
}

function setLoadingState(isLoading) {
    sendButton.disabled = isLoading;
    sendButton.classList.toggle("is-loading", isLoading);
    sendButton.textContent = isLoading ? "Sending..." : "Send";
    if (isLoading) {
        userInput.setAttribute("aria-disabled", "true");
    } else {
        userInput.removeAttribute("aria-disabled");
    }
}

function scrollToLatestMessage() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}