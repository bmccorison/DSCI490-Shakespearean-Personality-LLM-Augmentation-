import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const DEFAULT_WORK = "Hamlet";
const STARTUP_RETRY_ATTEMPTS = 20;
const STARTUP_RETRY_DELAY_MS = 1000;
// TODO: Replace this static list with backend-provided character options.
const CHARACTER_OPTIONS = ["Hamlet"];
const MULTIMODEL_CHARACTER_DEFAULTS = ["Hamlet", "Ophelia", "Macbeth", "Lady Macbeth"];
const MULTIMODEL_MIN_SPEAKERS = 2;
const MULTIMODEL_MAX_SPEAKERS = 4;
const MULTIMODEL_DEFAULT_MAX_TURNS = 12;
const MULTIMODEL_HARD_MAX_TURNS = 20;

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
    throw new Error(await getErrorMessage(response, path));
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

async function apiPostJson(path, payload = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await getErrorMessage(response, path));
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
    throw new Error(await getErrorMessage(response, path));
  }
  return response.blob();
}

async function getErrorMessage(response, path) {
  const fallbackMessage = `${path} failed (${response.status})`;
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    try {
      const payload = await response.json();
      if (payload && typeof payload.detail === "string") {
        return payload.detail;
      }
      if (payload && typeof payload.message === "string") {
        return payload.message;
      }
    } catch {
      return fallbackMessage;
    }
  }

  try {
    const text = await response.text();
    return text || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
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

function normalizeModels(payload) {
  if (!Array.isArray(payload)) return [];

  return payload
    .map((model) => {
      if (!model || typeof model.name !== "string") {
        return null;
      }

      const nextAdapters = Array.isArray(model.adapters)
        ? model.adapters
        : Array.isArray(model.adapter_paths)
          ? model.adapter_paths.map((adapter) => {
              if (!adapter || typeof adapter !== "object") {
                return null;
              }

              const pair = Object.entries(adapter).find(
                ([key, value]) => key !== "description" && typeof value === "string"
              );
              if (!pair) {
                return null;
              }

              const [name, path] = pair;
              return {
                name,
                path,
                description:
                  typeof adapter.description === "string" ? adapter.description : "",
              };
            })
          : [];

      const adapters = nextAdapters.filter(
        (adapter) =>
          adapter &&
          typeof adapter.name === "string" &&
          typeof adapter.path === "string" &&
          adapter.path.length > 0
      );

      return {
        name: model.name,
        description:
          typeof model.description === "string" ? model.description : "",
        defaultAdapterPath:
          typeof model.default_adapter_path === "string"
            ? model.default_adapter_path
            : "",
        adapters,
      };
    })
    .filter((model) => model && model.adapters.length > 0);
}

function resolveDefaultAdapterPath(model) {
  if (!model || !Array.isArray(model.adapters) || model.adapters.length === 0) {
    return "";
  }

  const preferredAdapter = model.adapters.find(
    (adapter) => adapter.path === model.defaultAdapterPath
  );
  return preferredAdapter?.path || model.adapters[0].path;
}

function createMultiModelParticipant(index, modelList = []) {
  const defaultModel = modelList[0] ?? null;
  return {
    name: `Speaker ${index + 1}`,
    character:
      MULTIMODEL_CHARACTER_DEFAULTS[index] || `Character ${index + 1}`,
    work: DEFAULT_WORK,
    model_name: defaultModel?.name || "",
    adapter_path: resolveDefaultAdapterPath(defaultModel),
  };
}

function normalizeMultiModelConfig(payload) {
  const hardMaxTurns =
    Number(payload?.hard_max_turns) || MULTIMODEL_HARD_MAX_TURNS;
  const defaultMaxTurns =
    Number(payload?.default_max_turns) || MULTIMODEL_DEFAULT_MAX_TURNS;
  return {
    defaultMaxTurns: Math.min(Math.max(1, defaultMaxTurns), hardMaxTurns),
    hardMaxTurns,
    minParticipants:
      Number(payload?.min_participants) || MULTIMODEL_MIN_SPEAKERS,
    maxParticipants:
      Number(payload?.max_participants) || MULTIMODEL_MAX_SPEAKERS,
  };
}

function formatTimestamp(date = new Date()) {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function isRetryableStartupError(error) {
  const message = error?.message || "";
  return (
    message.length === 0 ||
    /ECONNREFUSED|Failed to fetch|NetworkError|http proxy error|failed \(500\)|failed \(502\)|failed \(503\)|failed \(504\)/i.test(
      message
    )
  );
}

async function retryStartupAction(action, options = {}) {
  const { isCancelled, onRetry } = options;
  let lastError = null;

  for (let attempt = 1; attempt <= STARTUP_RETRY_ATTEMPTS; attempt += 1) {
    if (isCancelled?.()) {
      return null;
    }

    try {
      return await action();
    } catch (error) {
      lastError = error;
      if (
        attempt === STARTUP_RETRY_ATTEMPTS ||
        !isRetryableStartupError(error) ||
        isCancelled?.()
      ) {
        throw error;
      }

      onRetry?.(attempt + 1, error);
      await sleep(STARTUP_RETRY_DELAY_MS);
    }
  }

  throw lastError;
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
  const [isApplyingModel, setIsApplyingModel] = useState(false);
  const [speakingId, setSpeakingId] = useState(null);
  const [isAudioLoading, setIsAudioLoading] = useState(false);
  const [isAudioPaused, setIsAudioPaused] = useState(false);
  const [isShakespeareStyleEnabled, setIsShakespeareStyleEnabled] = useState(false);
  const [multiModelConfig, setMultiModelConfig] = useState({
    defaultMaxTurns: MULTIMODEL_DEFAULT_MAX_TURNS,
    hardMaxTurns: MULTIMODEL_HARD_MAX_TURNS,
    minParticipants: MULTIMODEL_MIN_SPEAKERS,
    maxParticipants: MULTIMODEL_MAX_SPEAKERS,
  });
  const [multiInitialPrompt, setMultiInitialPrompt] = useState(
    "Debate whether action or patience better serves Denmark."
  );
  const [multiMaxTurns, setMultiMaxTurns] = useState(MULTIMODEL_DEFAULT_MAX_TURNS);
  const [multiSpeakerCount, setMultiSpeakerCount] = useState(MULTIMODEL_MIN_SPEAKERS);
  const [multiParticipants, setMultiParticipants] = useState(() =>
    Array.from({ length: MULTIMODEL_MIN_SPEAKERS }, (_, index) =>
      createMultiModelParticipant(index)
    )
  );
  const [multiTurns, setMultiTurns] = useState([]);
  const [multiStatus, setMultiStatus] = useState("Configure speakers to begin.");
  const [multiError, setMultiError] = useState("");
  const [isMultiRunning, setIsMultiRunning] = useState(false);
  const [activityLog, setActivityLog] = useState([]);
  const bottomRef = useRef(null);
  const activeAudioRef = useRef(null);
  const activeAudioUrlRef = useRef("");
  const pendingModelApplyCountRef = useRef(0);
  const multiStopRequestedRef = useRef(false);

  const modelDetails = useMemo(
    () => models.find((model) => model.name === selectedModel),
    [models, selectedModel]
  );
  const adapterOptions = useMemo(
    () => modelDetails?.adapters ?? [],
    [modelDetails]
  );
  const selectedAdapterDetails = useMemo(
    () => adapterOptions.find((adapter) => adapter.path === selectedAdapter) ?? null,
    [adapterOptions, selectedAdapter]
  );
  const visibleMultiParticipants = useMemo(
    () => multiParticipants.slice(0, multiSpeakerCount),
    [multiParticipants, multiSpeakerCount]
  );

  useEffect(() => {
    if (adapterOptions.length === 0) {
      setSelectedAdapter("");
      return;
    }
    if (!adapterOptions.some((item) => item.path === selectedAdapter)) {
      setSelectedAdapter(adapterOptions[0].path);
    }
  }, [adapterOptions, selectedAdapter]);

  useEffect(() => {
    if (models.length === 0) {
      return;
    }

    setMultiParticipants((previous) =>
      previous.map((participant, index) => {
        const currentModel = models.find(
          (model) => model.name === participant.model_name
        );
        const modelDetailsForParticipant = currentModel ?? models[0];
        const adapterStillValid = modelDetailsForParticipant.adapters.some(
          (adapter) => adapter.path === participant.adapter_path
        );

        return {
          ...participant,
          model_name: modelDetailsForParticipant.name,
          adapter_path: adapterStillValid
            ? participant.adapter_path
            : resolveDefaultAdapterPath(modelDetailsForParticipant),
          name: participant.name || `Speaker ${index + 1}`,
        };
      })
    );
  }, [models]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  const releaseActiveAudio = () => {
    const activeAudio = activeAudioRef.current;
    if (activeAudio) {
      activeAudio.onended = null;
      activeAudio.onerror = null;
      activeAudio.onpause = null;
      activeAudio.onplay = null;
      activeAudio.pause();
      activeAudioRef.current = null;
    }

    if (activeAudioUrlRef.current) {
      URL.revokeObjectURL(activeAudioUrlRef.current);
      activeAudioUrlRef.current = "";
    }
  };

  const clearPlaybackState = () => {
    setSpeakingId(null);
    setIsAudioLoading(false);
    setIsAudioPaused(false);
  };

  useEffect(() => {
    return () => {
      releaseActiveAudio();
    };
  }, []);

  const recordActivity = (kind, detail) => {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      kind,
      detail,
      timestamp: formatTimestamp(),
    };

    setActivityLog((previous) => [entry, ...previous].slice(0, 12));
  };

  const updateStatus = (nextStatus, kind = "status") => {
    setStatus(nextStatus);
    recordActivity(kind, nextStatus);
  };

  const reportError = (message, fallback = "Action failed.") => {
    const nextError = message || fallback;
    setError(nextError);
    recordActivity("error", nextError);
  };

  const refreshServerChat = async (showStatus = true) => {
    setError("");
    releaseActiveAudio();
    clearPlaybackState();
    try {
      await apiGet("/refresh_chat");
    } catch (refreshError) {
      await apiGet("/select_character", {
        character,
        work: DEFAULT_WORK,
      });
      recordActivity(
        "refresh",
        `Refresh fallback used after /refresh_chat failed: ${refreshError.message}`
      );
    }
    setMessages([]);
    if (showStatus) {
      updateStatus("Chat history refreshed.", "refresh");
    }
  };

  const fetchModels = async (showStatus = true) => {
    setError("");
    const payload = await apiGet("/get_models");
    const modelList = normalizeModels(payload);
    setModels(modelList);

    if (modelList.length === 0) {
      setSelectedModel("");
      setSelectedAdapter("");
      if (showStatus) {
        updateStatus("No loadable models are currently available.", "models");
      }
      return modelList;
    }

    if (!modelList.some((model) => model.name === selectedModel)) {
      setSelectedModel(modelList[0].name);
      const defaultAdapter = resolveDefaultAdapterPath(modelList[0]);
      setSelectedAdapter(defaultAdapter);
    }

    if (showStatus) {
      updateStatus(`Loaded ${modelList.length} model option(s).`, "models");
    }
    return modelList;
  };

  const fetchMultiModelConfig = async () => {
    const payload = await apiGet("/multimodel/config");
    const config = normalizeMultiModelConfig(payload);
    setMultiModelConfig(config);
    setMultiMaxTurns(config.defaultMaxTurns);
    return config;
  };

  const saveMultiModelConfig = async (nextMaxTurns) => {
    const payload = await apiPostJson("/multimodel/config", {
      max_turns: nextMaxTurns,
    });
    const config = normalizeMultiModelConfig(payload);
    setMultiModelConfig(config);
    setMultiMaxTurns(config.defaultMaxTurns);
    return config;
  };

  const applyCharacter = async (nextCharacter = character, showStatus = true) => {
    setError("");
    releaseActiveAudio();
    clearPlaybackState();
    await apiGet("/select_character", {
      character: nextCharacter,
      work: DEFAULT_WORK,
    });
    setMessages([]);
    if (showStatus) {
      updateStatus(`Character set to ${nextCharacter}.`, "character");
    }
  };

  const applyModel = async (
    nextModel = selectedModel,
    nextAdapter = selectedAdapter,
    showStatus = true
  ) => {
    setError("");
    if (!nextModel || !nextAdapter) {
      throw new Error("Select a valid model and adapter before loading.");
    }

    const activeModel = models.find((model) => model.name === nextModel);
    const activeAdapter = activeModel?.adapters.find(
      (adapter) => adapter.path === nextAdapter
    );

    pendingModelApplyCountRef.current += 1;
    setIsApplyingModel(true);
    try {
      await apiGet("/select_model", {
        model_name: nextModel,
        adapter_path: nextAdapter,
      });
      if (showStatus) {
        updateStatus(
          `Model selection submitted: ${nextModel} with ${
            activeAdapter?.name || nextAdapter
          }.`,
          "model"
        );
      }
    } finally {
      pendingModelApplyCountRef.current = Math.max(
        0,
        pendingModelApplyCountRef.current - 1
      );
      if (pendingModelApplyCountRef.current === 0) {
        setIsApplyingModel(false);
      }
    }
  };

  const handleModelChange = (nextModel) => {
    setSelectedModel(nextModel);
    const nextModelDetails = models.find((model) => model.name === nextModel);
    const nextAdapter = resolveDefaultAdapterPath(nextModelDetails);
    setSelectedAdapter(nextAdapter);
    if (!nextAdapter) {
      reportError("No loadable adapter is available for that model.");
      return;
    }

    applyModel(nextModel, nextAdapter).catch((applyError) =>
      reportError(applyError.message, "Model apply failed.")
    );
  };

  const handleAdapterChange = (nextAdapter) => {
    setSelectedAdapter(nextAdapter);
    applyModel(selectedModel, nextAdapter).catch((applyError) =>
      reportError(applyError.message, "Model apply failed.")
    );
  };

  const handleCharacterChange = (nextCharacter) => {
    setCharacter(nextCharacter);
    applyCharacter(nextCharacter).catch((characterError) =>
      reportError(characterError.message, "Character update failed.")
    );
  };

  const handleStyleToggle = () => {
    setIsShakespeareStyleEnabled((isEnabled) => {
      const nextValue = !isEnabled;
      updateStatus(
        nextValue
          ? "Shakespeare dialogue polish enabled."
          : "Shakespeare dialogue polish disabled.",
        "style"
      );
      return nextValue;
    });
  };

  const parseMultiMaxTurns = () => {
    const parsedTurns = Number(multiMaxTurns);
    if (!Number.isFinite(parsedTurns)) {
      throw new Error("Model conversation turn limit must be a number.");
    }
    const wholeTurns = Math.floor(parsedTurns);
    if (wholeTurns < 1 || wholeTurns > multiModelConfig.hardMaxTurns) {
      throw new Error(
        `Turn limit must be between 1 and ${multiModelConfig.hardMaxTurns}.`
      );
    }
    return wholeTurns;
  };

  const handleSaveMultiMaxTurns = async () => {
    setMultiError("");
    try {
      const savedConfig = await saveMultiModelConfig(parseMultiMaxTurns());
      setMultiStatus(
        `Default model conversation limit saved at ${savedConfig.defaultMaxTurns} turn(s).`
      );
      recordActivity(
        "multimodel",
        `Default model conversation limit saved: ${savedConfig.defaultMaxTurns}.`
      );
    } catch (configError) {
      setMultiError(configError.message);
      recordActivity("error", configError.message);
    }
  };

  const updateMultiParticipant = (index, updates) => {
    setMultiParticipants((previous) =>
      previous.map((participant, participantIndex) =>
        participantIndex === index ? { ...participant, ...updates } : participant
      )
    );
  };

  const handleMultiModelChange = (index, nextModelName) => {
    const nextModelDetails = models.find((model) => model.name === nextModelName);
    updateMultiParticipant(index, {
      model_name: nextModelName,
      adapter_path: resolveDefaultAdapterPath(nextModelDetails),
    });
  };

  const handleMultiSpeakerCountChange = (nextCountValue) => {
    const parsedCount = Number(nextCountValue);
    const boundedCount = Math.min(
      Math.max(parsedCount, multiModelConfig.minParticipants),
      multiModelConfig.maxParticipants
    );
    setMultiSpeakerCount(boundedCount);
    setMultiParticipants((previous) => {
      const nextParticipants = [...previous];
      while (nextParticipants.length < boundedCount) {
        nextParticipants.push(
          createMultiModelParticipant(nextParticipants.length, models)
        );
      }
      return nextParticipants;
    });
  };

  const buildMultiStartPayload = () => {
    const initialPrompt = multiInitialPrompt.trim();
    if (!initialPrompt) {
      throw new Error("Enter an initial prompt for the model conversation.");
    }

    const participants = visibleMultiParticipants.map((participant) => ({
      name: participant.name.trim(),
      character: participant.character.trim(),
      work: participant.work.trim() || DEFAULT_WORK,
      model_name: participant.model_name.trim(),
      adapter_path: participant.adapter_path.trim(),
    }));
    const incompleteParticipant = participants.find(
      (participant) =>
        !participant.name ||
        !participant.character ||
        !participant.model_name ||
        !participant.adapter_path
    );
    if (incompleteParticipant) {
      throw new Error("Each speaker needs a name, character, model, and adapter.");
    }

    return {
      initial_prompt: initialPrompt,
      max_turns: parseMultiMaxTurns(),
      shakespeare_style: isShakespeareStyleEnabled,
      participants,
    };
  };

  const handleStartMultiConversation = async () => {
    if (isMultiRunning || isSending || isApplyingModel) {
      return;
    }

    let startPayload;
    try {
      startPayload = buildMultiStartPayload();
    } catch (validationError) {
      setMultiError(validationError.message);
      return;
    }

    setMultiError("");
    setMultiTurns([]);
    setIsMultiRunning(true);
    multiStopRequestedRef.current = false;
    releaseActiveAudio();
    clearPlaybackState();
    setMultiStatus("Starting model conversation...");
    recordActivity("multimodel", "Model conversation started.");

    try {
      await saveMultiModelConfig(startPayload.max_turns);
      let session = await apiPostJson("/multimodel/start", startPayload);
      setMultiTurns(Array.isArray(session.turns) ? session.turns : []);

      while (
        !multiStopRequestedRef.current &&
        session?.status === "running" &&
        session.turn_count < startPayload.max_turns
      ) {
        const nextSpeakerName = session.next_speaker?.name || "Next speaker";
        setMultiStatus(`${nextSpeakerName} is composing...`);
        session = await apiPostJson("/multimodel/next");
        if (Array.isArray(session.turns)) {
          setMultiTurns(session.turns);
        }
        if (session.last_turn?.speaker_name) {
          recordActivity(
            "multimodel",
            `${session.last_turn.speaker_name} added turn ${session.last_turn.turn_number}.`
          );
        }
      }

      if (multiStopRequestedRef.current || session?.status === "stopped") {
        setMultiStatus("Model conversation stopped.");
      } else {
        setMultiStatus(
          `Model conversation complete at ${session?.turn_count || 0} turn(s).`
        );
      }
    } catch (conversationError) {
      setMultiError(conversationError.message);
      recordActivity("error", conversationError.message);
      setMultiStatus("Model conversation failed.");
    } finally {
      setIsMultiRunning(false);
    }
  };

  const handleStopMultiConversation = async () => {
    multiStopRequestedRef.current = true;
    setMultiStatus("Stopping after the current turn...");
    try {
      const session = await apiPostJson("/multimodel/stop");
      if (Array.isArray(session.turns)) {
        setMultiTurns(session.turns);
      }
    } catch (stopError) {
      setMultiError(stopError.message);
      recordActivity("error", stopError.message);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      setError("");
      updateStatus("Preparing thy stage...", "startup");
      try {
        await retryStartupAction(() => applyCharacter("Hamlet", false), {
          isCancelled: () => cancelled,
          onRetry: (nextAttempt) => {
            setStatus(
              `Waiting for backend to start... retry ${nextAttempt}/${STARTUP_RETRY_ATTEMPTS}.`
            );
          },
        });
        if (cancelled) return;
        recordActivity("character", "Default character context applied.");

        try {
          const config = await fetchMultiModelConfig();
          recordActivity(
            "multimodel",
            `Model conversation limit set to ${config.defaultMaxTurns} turn(s).`
          );
        } catch (configError) {
          recordActivity(
            "multimodel",
            `Using default model conversation limit after config failed: ${configError.message}`
          );
        }

        const loadedModels = await retryStartupAction(() => fetchModels(false), {
          isCancelled: () => cancelled,
          onRetry: (nextAttempt) => {
            setStatus(
              `Backend reached. Loading models... retry ${nextAttempt}/${STARTUP_RETRY_ATTEMPTS}.`
            );
          },
        });
        if (cancelled) return;
        recordActivity(
          "models",
          `Discovered ${loadedModels.length} loadable model option(s).`
        );

        const firstModel = loadedModels[0];
        const firstAdapter = resolveDefaultAdapterPath(firstModel);
        if (firstModel?.name && firstAdapter) {
          setSelectedModel(firstModel.name);
          setSelectedAdapter(firstAdapter);
          await applyModel(firstModel.name, firstAdapter, false);
          recordActivity(
            "model",
            `Default model loaded: ${firstModel.name} using ${firstAdapter}.`
          );
        }

        if (!cancelled) {
          updateStatus(
            firstModel?.name && firstAdapter
              ? "Thy chatbot is ready."
              : "No loadable models are currently available.",
            "startup"
          );
        }
      } catch (initError) {
        if (!cancelled) {
          reportError(initError.message, "Could not initialize interface.");
          updateStatus("Initialization finished with warnings.", "startup");
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
    if (!question || isSending || isApplyingModel || isMultiRunning) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
    };

    setMessages((previous) => [...previous, userMessage]);
    setDraft("");
    setIsSending(true);
    setError("");
    recordActivity("message", `Prompt sent: ${question}`);
    updateStatus("Hamlet is composing a reply...", "generation");

    try {
      const payload = await apiGet("/generate_response", {
        question,
        shakespeare_style: isShakespeareStyleEnabled,
      });
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
      updateStatus("A reply hath arrived.", "reply");
    } catch (sendError) {
      reportError(sendError.message, "Message send failed.");
      updateStatus("Reply failed.", "error");
    } finally {
      setIsSending(false);
    }
  };

  const handleSpeak = async (messageId, text) => {
    setError("");
    releaseActiveAudio();
    setSpeakingId(messageId);
    setIsAudioLoading(true);
    setIsAudioPaused(false);
    updateStatus("Preparing spoken performance...", "speech");
    try {
      const audioBlob = await apiPostBlob("/tts", { text, character });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      activeAudioRef.current = audio;
      activeAudioUrlRef.current = audioUrl;

      audio.onended = () => {
        releaseActiveAudio();
        clearPlaybackState();
      };
      audio.onerror = () => {
        releaseActiveAudio();
        clearPlaybackState();
      };
      audio.onpause = () => {
        if (!audio.ended) {
          setIsAudioPaused(true);
        }
      };
      audio.onplay = () => setIsAudioPaused(false);
      await audio.play();
      updateStatus("Thy line is now spoken aloud.", "speech");
    } catch (ttsError) {
      releaseActiveAudio();
      clearPlaybackState();
      reportError(ttsError.message, "Could not generate speech.");
    }
    setIsAudioLoading(false);
  };

  const handlePauseResume = async () => {
    const activeAudio = activeAudioRef.current;
    if (!activeAudio || speakingId === null) {
      return;
    }

    if (activeAudio.paused) {
      try {
        await activeAudio.play();
        setIsAudioPaused(false);
        updateStatus("Speech resumed.", "speech");
      } catch (resumeError) {
        reportError(resumeError.message, "Could not resume speech.");
      }
      return;
    }

    activeAudio.pause();
    setIsAudioPaused(true);
    updateStatus("Speech paused.", "speech");
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
                disabled={models.length === 0 || isMultiRunning}
              >
                {models.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name}
                  </option>
                ))}
                {models.length === 0 && <option>No models available</option>}
              </select>
              <p className="mt-1 min-h-10 text-sm text-maroon/75">
                {modelDetails?.description || "No model description available."}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-maroon">
                Adapter
              </label>
              <select
                className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                value={selectedAdapter}
                onChange={(event) => handleAdapterChange(event.target.value)}
                disabled={adapterOptions.length === 0 || isMultiRunning}
              >
                {adapterOptions.map((adapter) => (
                  <option key={adapter.path} value={adapter.path}>
                    {adapter.name}
                  </option>
                ))}
                {adapterOptions.length === 0 && <option>No adapter</option>}
              </select>
              <p className="mt-1 min-h-10 text-sm text-maroon/75">
                {selectedAdapterDetails?.description ||
                  "No adapter description available."}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-maroon">
                Character
              </label>
              <select
                className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                value={character}
                onChange={(event) => handleCharacterChange(event.target.value)}
                disabled={isMultiRunning}
              >
                {CHARACTER_OPTIONS.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-maroon/80">
              Selections apply automatically.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <button
                className={`rounded-lg border px-3 py-2 text-sm font-semibold ${
                  isShakespeareStyleEnabled
                    ? "border-maroon bg-maroon text-white"
                    : "border-maroon bg-white text-maroon"
                }`}
                onClick={handleStyleToggle}
                type="button"
                aria-pressed={isShakespeareStyleEnabled}
                disabled={isMultiRunning}
              >
                Shakespeare Style: {isShakespeareStyleEnabled ? "On" : "Off"}
              </button>
              <button
                className="rounded-lg border border-maroon bg-white px-3 py-2 text-sm font-semibold text-maroon"
                onClick={() =>
                  refreshServerChat().catch((refreshError) =>
                    reportError(refreshError.message, "Chat reset failed.")
                  )
                }
                type="button"
                disabled={isMultiRunning}
              >
                Refresh Chat
              </button>
            </div>
          </div>
        </details>
      </section>

      <section className="mt-4 rounded-2xl border-2 border-maroon bg-white p-4 shadow-[0_8px_24px_rgba(165,46,48,0.1)]">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-maroon">
              Model Conversation
            </h2>
            <p className="mt-1 text-sm text-maroon/75">
              Configure two to four speakers, then let them answer each other in order.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              className="rounded-lg border border-maroon bg-maroon px-3 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStartMultiConversation}
              type="button"
              disabled={
                isMultiRunning ||
                isSending ||
                isApplyingModel ||
                models.length === 0
              }
            >
              {isMultiRunning ? "Running" : "Start Conversation"}
            </button>
            <button
              className="rounded-lg border border-maroon bg-white px-3 py-2 text-sm font-semibold text-maroon disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStopMultiConversation}
              type="button"
              disabled={!isMultiRunning}
            >
              Stop
            </button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_13rem_12rem]">
          <label className="block">
            <span className="text-sm font-medium text-maroon">
              Initial prompt
            </span>
            <textarea
              className="mt-1 min-h-24 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon placeholder:text-maroon/50"
              value={multiInitialPrompt}
              onChange={(event) => setMultiInitialPrompt(event.target.value)}
              disabled={isMultiRunning}
            />
          </label>

          <label className="block">
            <span className="text-sm font-medium text-maroon">
              Speakers
            </span>
            <select
              className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
              value={multiSpeakerCount}
              onChange={(event) =>
                handleMultiSpeakerCountChange(event.target.value)
              }
              disabled={isMultiRunning}
            >
              {Array.from(
                {
                  length:
                    multiModelConfig.maxParticipants -
                    multiModelConfig.minParticipants +
                    1,
                },
                (_, index) => multiModelConfig.minParticipants + index
              ).map((count) => (
                <option key={count} value={count}>
                  {count}
                </option>
              ))}
            </select>
          </label>

          <div>
            <label className="block">
              <span className="text-sm font-medium text-maroon">
                Max turns
              </span>
              <input
                className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                type="number"
                min="1"
                max={multiModelConfig.hardMaxTurns}
                value={multiMaxTurns}
                onChange={(event) => setMultiMaxTurns(event.target.value)}
                disabled={isMultiRunning}
              />
            </label>
            <button
              className="mt-2 w-full rounded-lg border border-maroon bg-white px-3 py-2 text-sm font-semibold text-maroon disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleSaveMultiMaxTurns}
              type="button"
              disabled={isMultiRunning}
            >
              Save Limit
            </button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {visibleMultiParticipants.map((participant, index) => {
            const participantModel = models.find(
              (model) => model.name === participant.model_name
            );
            const participantAdapters = participantModel?.adapters ?? [];

            return (
              <article
                key={`${index}-${participant.name}`}
                className="rounded-xl border border-maroon/20 bg-parchment p-3"
              >
                <div className="grid gap-2 sm:grid-cols-3">
                  <label className="block">
                    <span className="text-sm font-medium text-maroon">
                      Speaker
                    </span>
                    <input
                      className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                      value={participant.name}
                      onChange={(event) =>
                        updateMultiParticipant(index, {
                          name: event.target.value,
                        })
                      }
                      disabled={isMultiRunning}
                    />
                  </label>
                  <label className="block">
                    <span className="text-sm font-medium text-maroon">
                      Character
                    </span>
                    <input
                      className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                      value={participant.character}
                      onChange={(event) =>
                        updateMultiParticipant(index, {
                          character: event.target.value,
                        })
                      }
                      disabled={isMultiRunning}
                    />
                  </label>
                  <label className="block">
                    <span className="text-sm font-medium text-maroon">
                      Work
                    </span>
                    <input
                      className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                      value={participant.work}
                      onChange={(event) =>
                        updateMultiParticipant(index, {
                          work: event.target.value,
                        })
                      }
                      disabled={isMultiRunning}
                    />
                  </label>
                </div>

                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  <label className="block">
                    <span className="text-sm font-medium text-maroon">
                      Model
                    </span>
                    <select
                      className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                      value={participant.model_name}
                      onChange={(event) =>
                        handleMultiModelChange(index, event.target.value)
                      }
                      disabled={isMultiRunning || models.length === 0}
                    >
                      {models.map((model) => (
                        <option key={model.name} value={model.name}>
                          {model.name}
                        </option>
                      ))}
                      {models.length === 0 && <option>No models available</option>}
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-sm font-medium text-maroon">
                      Adapter
                    </span>
                    <select
                      className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                      value={participant.adapter_path}
                      onChange={(event) =>
                        updateMultiParticipant(index, {
                          adapter_path: event.target.value,
                        })
                      }
                      disabled={isMultiRunning || participantAdapters.length === 0}
                    >
                      {participantAdapters.map((adapter) => (
                        <option key={adapter.path} value={adapter.path}>
                          {adapter.name}
                        </option>
                      ))}
                      {participantAdapters.length === 0 && <option>No adapter</option>}
                    </select>
                  </label>
                </div>
              </article>
            );
          })}
        </div>

        <div className="mt-4 grid gap-3 lg:grid-cols-[16rem_minmax(0,1fr)]">
          <aside className="rounded-xl border border-maroon/20 bg-parchment p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-maroon/60">
              Conversation Status
            </p>
            <p className="mt-2 text-base text-maroon">{multiStatus}</p>
            {multiError && (
              <p className="mt-3 rounded-lg border border-maroon/20 bg-white px-3 py-2 text-sm text-maroon">
                {multiError}
              </p>
            )}
          </aside>

          <div className="max-h-80 overflow-y-auto rounded-xl border border-maroon/20 bg-parchment p-3">
            {multiTurns.length === 0 && (
              <p className="py-8 text-center text-base text-maroon/70">
                Generated model turns will appear here.
              </p>
            )}
            {multiTurns.map((turn) => (
              <article
                key={`${turn.turn_number}-${turn.speaker_name}`}
                className="mb-3 rounded-xl border border-gold bg-white px-4 py-3 text-maroon last:mb-0"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-semibold">
                    {turn.speaker_name} as {turn.character}
                  </p>
                  <span className="text-sm text-maroon/60">
                    Turn {turn.turn_number}
                  </span>
                </div>
                <p className="mt-2 whitespace-pre-wrap text-lg leading-relaxed">
                  {turn.content}
                </p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <article className="rounded-2xl border border-maroon/20 bg-white px-4 py-4 shadow-[0_6px_18px_rgba(69,20,21,0.08)]">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-maroon/60">
            Current Status
          </p>
          <p className="mt-2 text-lg text-maroon">{status}</p>
          {error && (
            <p className="mt-3 rounded-xl border border-maroon/20 bg-maroon/5 px-3 py-2 text-sm text-maroon">
              {error}
            </p>
          )}
        </article>

        <article className="rounded-2xl border border-maroon/20 bg-white px-4 py-4 shadow-[0_6px_18px_rgba(69,20,21,0.08)]">
          <div className="flex items-center justify-between gap-3">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-maroon/60">
              Activity Log
            </p>
            <span className="text-xs text-maroon/60">
              {activityLog.length} recent event{activityLog.length === 1 ? "" : "s"}
            </span>
          </div>
          <div className="mt-3 max-h-40 space-y-2 overflow-y-auto pr-1">
            {activityLog.length === 0 && (
              <p className="rounded-xl border border-dashed border-maroon/20 bg-parchment px-3 py-3 text-sm text-maroon/70">
                Actions will appear here as the interface works.
              </p>
            )}
            {activityLog.map((entry) => (
              <div
                key={entry.id}
                className={`rounded-xl border px-3 py-2 text-sm ${
                  entry.kind === "error"
                    ? "border-maroon/30 bg-maroon/5 text-maroon"
                    : "border-gold/50 bg-parchment text-maroon"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="font-semibold capitalize">{entry.kind}</span>
                  <span className="text-xs text-maroon/60">{entry.timestamp}</span>
                </div>
                <p className="mt-1 leading-snug">{entry.detail}</p>
              </div>
            ))}
          </div>
        </article>
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
                  <div className="mt-2 flex flex-wrap gap-2">
                    <button
                      className="rounded-md border border-maroon px-2 py-1 text-sm font-medium text-maroon hover:bg-gold disabled:cursor-not-allowed disabled:opacity-60"
                      onClick={() => handleSpeak(message.id, message.content)}
                      type="button"
                      disabled={isAudioLoading && speakingId === message.id}
                    >
                      {isAudioLoading && speakingId === message.id
                        ? "Voicing..."
                        : speakingId === message.id && !isAudioPaused
                          ? "Playing..."
                          : "Play Voice"}
                    </button>
                    {speakingId === message.id && !isAudioLoading && (
                      <button
                        className="rounded-md border border-maroon px-2 py-1 text-sm font-medium text-maroon hover:bg-gold"
                        onClick={handlePauseResume}
                        type="button"
                      >
                        {isAudioPaused ? "Resume Voice" : "Pause Voice"}
                      </button>
                    )}
                  </div>
                )}
              </article>
            </div>
          ))}

          {isSending && (
            <div className="message-row mb-3 flex max-w-[96%] items-start gap-2">
              <div className="message-icon mt-1 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-gold bg-white text-maroon">
                <img src="/crown.svg" className="h-5 w-5" alt="" />
              </div>

              <article className="typing-indicator max-w-[92%] rounded-xl border border-gold bg-white px-4 py-3 text-maroon">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-semibold text-maroon/75">
                    Hamlet is drafting
                  </span>
                  <div className="flex items-center gap-1.5" aria-hidden="true">
                    <span className="typing-dot" />
                    <span className="typing-dot typing-dot-delay-1" />
                    <span className="typing-dot typing-dot-delay-2" />
                  </div>
                </div>
              </article>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <form className="mt-4 flex gap-2" onSubmit={handleSend}>
          <input
            className="flex-1 rounded-xl border border-maroon/35 bg-white px-4 py-3 text-lg text-maroon placeholder:text-maroon/50 focus:border-maroon focus:outline-none"
            placeholder="What sayest thou?"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            disabled={isSending || isApplyingModel || isMultiRunning}
          />
          <button
            type="submit"
            disabled={isSending || isApplyingModel || isMultiRunning}
            className="send-quill-btn inline-flex h-12 min-w-12 items-center justify-center rounded-lg border-2 border-gold bg-white px-3 shadow-sm transition hover:bg-gold/20 disabled:cursor-not-allowed disabled:opacity-60"
            aria-label="Send message"
          >
            {isSending ? (
              <span className="inline-flex items-center gap-2 text-sm font-semibold text-maroon">
                <span className="loading-ripple" aria-hidden="true" />
                Sending
              </span>
            ) : isApplyingModel ? (
              <span className="inline-flex items-center gap-2 text-sm font-semibold text-maroon">
                Applying
              </span>
            ) : isMultiRunning ? (
              <span className="inline-flex items-center gap-2 text-sm font-semibold text-maroon">
                Models
              </span>
            ) : (
              <img
                src="/quill.svg"
                alt=""
                className="h-7 w-7"
                aria-hidden="true"
              />
            )}
          </button>
        </form>
      </section>

      <footer className="mt-3 min-h-6 text-sm text-maroon/70">
        <p>
          Latest event: {activityLog[0]?.detail || status}
        </p>
      </footer>
    </div>
  );
}
