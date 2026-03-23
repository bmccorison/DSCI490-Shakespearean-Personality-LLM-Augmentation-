import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const DEFAULT_WORK = "Hamlet";
// TODO: Replace this static list with backend-provided character options.
const CHARACTER_OPTIONS = ["Hamlet"];

function RobotIcon({ className = "h-5 w-5" }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <rect
        x="4"
        y="7"
        width="16"
        height="12"
        rx="3"
        stroke="currentColor"
        strokeWidth="1.7"
      />
      <path
        d="M12 3v3"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <circle cx="9" cy="12" r="1.2" fill="currentColor" />
      <circle cx="15" cy="12" r="1.2" fill="currentColor" />
      <path
        d="M9 16h6"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

function toQuery(params = {}) {
  const cleanParams = Object.entries(params).reduce((acc, [key, value]) => {
    if (value !== undefined && value !== null) {
      acc[key] = String(value);
    }
    return acc;
  }, {});
  return new URLSearchParams(cleanParams).toString();
}

async function apiGet(path, params) {
  const queryString = toQuery(params);
  const response = await fetch(
    `${API_BASE}${path}${queryString ? `?${queryString}` : ""}`,
    {
      method: "GET",
    }
  );
  if (!response.ok) {
    throw new Error(`${path} failed (${response.status})`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

async function apiPostBlob(path, params) {
  const queryString = toQuery(params);
  const response = await fetch(
    `${API_BASE}${path}${queryString ? `?${queryString}` : ""}`,
    {
      method: "POST",
    }
  );
  if (!response.ok) {
    throw new Error(`${path} failed (${response.status})`);
  }
  return response.blob();
}

function parseAssistantReply(payload) {
  if (typeof payload === "string") return payload;
  if (payload && typeof payload === "object") {
    if (typeof payload.response === "string") return payload.response;
    if (typeof payload.answer === "string") return payload.answer;
    if (typeof payload.message === "string") return payload.message;
    return JSON.stringify(payload);
  }
  return "The stage is silent.";
}

function getAdapters(model) {
  if (!model || !Array.isArray(model.adapter_paths)) return [];
  return model.adapter_paths
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") return null;
      const pair = Object.entries(entry).find(
        ([key, value]) => key !== "description" && typeof value === "string"
      );
      if (!pair) return null;
      const [name, path] = pair;
      return {
        label: entry.description ? `${name} - ${entry.description}` : name,
        value: path,
        key: `${name}-${index}`,
      };
    })
    .filter(Boolean);
}

export default function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedAdapter, setSelectedAdapter] = useState("");
  const [character, setCharacter] = useState("Hamlet");
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState("Awaiting thy command.");
  const [error, setError] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [speakingId, setSpeakingId] = useState(null);
  const bottomRef = useRef(null);

  const modelDetails = useMemo(
    () => models.find((model) => model.name === selectedModel),
    [models, selectedModel]
  );
  const adapterOptions = useMemo(() => getAdapters(modelDetails), [modelDetails]);

  useEffect(() => {
    if (adapterOptions.length === 0) {
      setSelectedAdapter("");
      return;
    }
    if (!adapterOptions.some((item) => item.value === selectedAdapter)) {
      setSelectedAdapter(adapterOptions[0].value);
    }
  }, [adapterOptions, selectedAdapter]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const refreshServerChat = async (showStatus = true) => {
    await apiGet("/refresh_chat");
    setMessages([]);
    if (showStatus) {
      setStatus("Chat history refreshed.");
    }
  };

  const fetchModels = async (showStatus = true) => {
    const payload = await apiGet("/get_models");
    const modelList = Array.isArray(payload) ? payload : [];
    setModels(modelList);

    if (modelList.length > 0 && !selectedModel) {
      setSelectedModel(modelList[0].name);
      const defaultAdapter = getAdapters(modelList[0])[0]?.value || "";
      setSelectedAdapter(defaultAdapter);
    }

    if (showStatus) {
      setStatus(`Loaded ${modelList.length} model option(s).`);
    }
    return modelList;
  };

  const applyCharacter = async (nextCharacter = character, showStatus = true) => {
    await apiGet("/select_character", {
      character: nextCharacter,
      work: DEFAULT_WORK,
    });
    if (showStatus) {
      setStatus(`Character set to ${nextCharacter}.`);
    }
  };

  const applyModel = async (
    nextModel = selectedModel,
    nextAdapter = selectedAdapter,
    showStatus = true
  ) => {
    await apiGet("/select_model", {
      model_name: nextModel,
      adapter_path: nextAdapter,
    });
    if (showStatus) {
      setStatus(`Model selection submitted: ${nextModel}.`);
    }
  };

  const handleModelChange = (nextModel) => {
    setSelectedModel(nextModel);
    const nextModelDetails = models.find((model) => model.name === nextModel);
    const nextAdapter = getAdapters(nextModelDetails)[0]?.value || "";
    setSelectedAdapter(nextAdapter);
    applyModel(nextModel, nextAdapter).catch((applyError) =>
      setError(applyError.message || "Model apply failed.")
    );
  };

  const handleAdapterChange = (nextAdapter) => {
    setSelectedAdapter(nextAdapter);
    applyModel(selectedModel, nextAdapter).catch((applyError) =>
      setError(applyError.message || "Model apply failed.")
    );
  };

  const handleCharacterChange = (nextCharacter) => {
    setCharacter(nextCharacter);
    applyCharacter(nextCharacter).catch((characterError) =>
      setError(characterError.message || "Character update failed.")
    );
  };

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      setError("");
      setStatus("Preparing thy stage...");
      try {
        await refreshServerChat(false);
        if (cancelled) return;

        await applyCharacter("Hamlet", false);
        if (cancelled) return;

        const loadedModels = await fetchModels(false);
        if (cancelled) return;

        const firstModel = loadedModels[0];
        if (firstModel?.name) {
          const firstAdapter = getAdapters(firstModel)[0]?.value || "";
          setSelectedModel(firstModel.name);
          setSelectedAdapter(firstAdapter);
          await applyModel(firstModel.name, firstAdapter, false);
        }

        if (!cancelled) {
          setStatus("Thy chatbot is ready.");
        }
      } catch (initError) {
        if (!cancelled) {
          setError(initError.message || "Could not initialize interface.");
          setStatus("Initialization finished with warnings.");
        }
      }
    };

    initialize();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSend = async (event) => {
    event.preventDefault();
    const question = draft.trim();
    if (!question || isSending) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
    };

    setMessages((previous) => [...previous, userMessage]);
    setDraft("");
    setIsSending(true);
    setError("");

    try {
      const payload = await apiGet("/generate_response", { question });
      const answerText = parseAssistantReply(payload);
      const confidence =
        payload && typeof payload.confidence_score !== "undefined"
          ? `\n\nConfidence: ${payload.confidence_score}`
          : "";

      setMessages((previous) => [
        ...previous,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: `${answerText}${confidence}`,
        },
      ]);
      setStatus("A reply hath arrived.");
    } catch (sendError) {
      setError(sendError.message || "Message send failed.");
      setStatus("Reply failed.");
    } finally {
      setIsSending(false);
    }
  };

  const handleSpeak = async (messageId, text) => {
    setError("");
    setSpeakingId(messageId);
    try {
      const audioBlob = await apiPostBlob("/tts", { text, character });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.onended = () => URL.revokeObjectURL(audioUrl);
      audio.onerror = () => URL.revokeObjectURL(audioUrl);
      await audio.play();
      setStatus("Thy line is now spoken aloud.");
    } catch (ttsError) {
      setError(ttsError.message || "Could not generate speech.");
    } finally {
      setSpeakingId(null);
    }
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-5xl flex-col px-4 py-8 md:px-8">
      <header className="rounded-2xl border-2 border-maroon bg-white px-5 py-6 shadow-[0_10px_30px_rgba(165,46,48,0.16)]">
        <h1 className="break-words font-hamlet text-[clamp(1.6rem,5vw,3.4rem)] leading-tight text-maroon">
          Shakesperean Character Language Models
        </h1>
      </header>

      <section className="mt-6">
        <details className="rounded-2xl border border-maroon/25 bg-white p-4" open>
          <summary className="cursor-pointer font-semibold text-maroon">
            Controls
          </summary>

          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <div>
              <label className="block text-sm font-medium text-maroon">
                Model
              </label>
              <select
                className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                value={selectedModel}
                onChange={(event) => handleModelChange(event.target.value)}
              >
                {models.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name}
                  </option>
                ))}
                {models.length === 0 && <option>No models available</option>}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-maroon">
                Adapter
              </label>
              <select
                className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                value={selectedAdapter}
                onChange={(event) => handleAdapterChange(event.target.value)}
                disabled={adapterOptions.length === 0}
              >
                {adapterOptions.map((adapter) => (
                  <option key={adapter.key} value={adapter.value}>
                    {adapter.label}
                  </option>
                ))}
                {adapterOptions.length === 0 && <option>No adapter</option>}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-maroon">
                Character
              </label>
              <select
                className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                value={character}
                onChange={(event) => handleCharacterChange(event.target.value)}
              >
                {CHARACTER_OPTIONS.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-3 flex items-center justify-between gap-3">
            <p className="text-sm text-maroon/80">
              Selections apply automatically.
            </p>
            <button
              className="rounded-lg border border-maroon bg-white px-3 py-2 text-sm font-semibold text-maroon"
              onClick={() =>
                refreshServerChat().catch((refreshError) =>
                  setError(refreshError.message || "Chat reset failed.")
                )
              }
              type="button"
            >
              Refresh Chat
            </button>
          </div>
        </details>
      </section>

      <section className="mt-6 flex flex-1 flex-col rounded-2xl border-2 border-maroon bg-parchment p-4 shadow-[0_8px_24px_rgba(165,46,48,0.12)]">
        <div className="h-[360px] overflow-y-auto pr-2">
          {messages.length === 0 && (
            <p className="pt-10 text-center text-lg text-maroon/75">
              Speak to {character} to begin the conversation.
            </p>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`message-row mb-3 flex max-w-[96%] items-start gap-2 ${
                message.role === "user"
                  ? "ml-auto justify-end flex-row-reverse"
                  : "mr-auto"
              }`}
            >
              <div className="message-icon mt-1 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-gold bg-white text-maroon">
                {message.role === "user" ? (
                  <img
                    src="/quill.svg"
                    alt=""
                    className="h-5 w-5"
                    aria-hidden="true"
                  />
                ) : (
                  <RobotIcon className="h-5 w-5" />
                )}
              </div>

              <article
                className={`max-w-[92%] rounded-xl border px-4 py-3 ${
                  message.role === "user"
                    ? "border-maroon bg-maroon text-white"
                    : "border-gold bg-white text-maroon"
                }`}
              >
                <p className="whitespace-pre-wrap text-lg leading-relaxed">
                  {message.content}
                </p>
                {message.role === "assistant" && (
                  <button
                    className="mt-2 rounded-md border border-maroon px-2 py-1 text-sm font-medium text-maroon hover:bg-gold"
                    onClick={() => handleSpeak(message.id, message.content)}
                    type="button"
                    disabled={speakingId === message.id}
                  >
                    {speakingId === message.id ? "Voicing..." : "Play Voice"}
                  </button>
                )}
              </article>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <form className="mt-4 flex gap-2" onSubmit={handleSend}>
          <input
            className="flex-1 rounded-xl border border-maroon/35 bg-white px-4 py-3 text-lg text-maroon placeholder:text-maroon/50 focus:border-maroon focus:outline-none"
            placeholder="What sayest thou?"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            disabled={isSending}
          />
          <button
            type="submit"
            disabled={isSending}
            className="send-quill-btn inline-flex h-12 w-12 items-center justify-center rounded-lg border-2 border-gold bg-white shadow-sm transition hover:bg-gold/20 disabled:cursor-not-allowed disabled:opacity-60"
            aria-label="Send message"
          >
            <img
              src="/quill.svg"
              alt=""
              className="h-7 w-7"
              aria-hidden="true"
            />
          </button>
        </form>
      </section>

      <footer className="mt-3 min-h-6 text-sm text-maroon/85">
        <p>{status}</p>
        {error && <p className="text-red-700">{error}</p>}
      </footer>
    </div>
  );
}
