import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const DEFAULT_WORK = "Hamlet";
const STARTUP_RETRY_ATTEMPTS = 20;
const STARTUP_RETRY_DELAY_MS = 1000;
// Fallback while backend model metadata is loading.
const CHARACTER_OPTIONS = ["Hamlet"];
const MULTIMODEL_MIN_SPEAKERS = 2;
const MULTIMODEL_MAX_SPEAKERS = 4;
const MULTIMODEL_DEFAULT_MAX_TURNS = 12;
const MULTIMODEL_HARD_MAX_TURNS = 20;
const DEFAULT_VOICE_OPTION = "default";

function MessageAvatar({ type = "assistant" }) {
  const isUser = type === "user";
  return (
    <div className="message-icon message-avatar mt-1 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-gold bg-white text-maroon">
      <img
        src={isUser ? "/quill.svg" : "/crown.svg"}
        alt=""
        className={`${isUser ? "h-5 w-5" : "h-6 w-6"} object-contain`}
        aria-hidden="true"
      />
    </div>
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
    },
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
    },
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

function normalizeText(value, fallback = "") {
  if (typeof value === "string" && value.trim().length > 0) {
    return value.trim();
  }
  return fallback;
}

function normalizeModels(payload) {
  if (!Array.isArray(payload)) return [];

  return payload
    .map((model) => {
      if (!model || typeof model.name !== "string") {
        return null;
      }

      const modelCharacter = normalizeText(model.character, CHARACTER_OPTIONS[0]);
      const modelWork = normalizeText(model.work, DEFAULT_WORK);
      const nextAdapters = Array.isArray(model.adapters)
        ? model.adapters
        : Array.isArray(model.adapter_paths)
          ? model.adapter_paths.map((adapter) => {
              if (!adapter || typeof adapter !== "object") {
                return null;
              }

              const pair = Object.entries(adapter).find(
                ([key, value]) =>
                  key !== "description" && typeof value === "string",
              );
              if (!pair) {
                return null;
              }

              const [name, path] = pair;
              return {
                name,
                path,
                description:
                  typeof adapter.description === "string"
                    ? adapter.description
                    : "",
                character: normalizeText(adapter.character, modelCharacter),
                work: normalizeText(adapter.work, modelWork),
              };
            })
          : [];

      const adapters = nextAdapters
        .filter(
          (adapter) =>
            adapter &&
            typeof adapter.name === "string" &&
            typeof adapter.path === "string" &&
            adapter.path.length > 0,
        )
        .map((adapter) => ({
          name: adapter.name,
          path: adapter.path,
          description:
            typeof adapter.description === "string"
              ? adapter.description
              : "",
          character: normalizeText(adapter.character, modelCharacter),
          work: normalizeText(adapter.work, modelWork),
        }));

      return {
        name: model.name,
        description:
          typeof model.description === "string" ? model.description : "",
        character: modelCharacter,
        work: modelWork,
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
    (adapter) => adapter.path === model.defaultAdapterPath,
  );
  return preferredAdapter?.path || model.adapters[0].path;
}

function resolveAdapterDetails(model, adapterPath) {
  return (
    model?.adapters?.find((adapter) => adapter.path === adapterPath) ?? null
  );
}

function resolveParticipantContext(model, adapterPath) {
  const adapter = resolveAdapterDetails(model, adapterPath);
  const modelCharacter = normalizeText(model?.character, CHARACTER_OPTIONS[0]);
  const modelWork = normalizeText(model?.work, DEFAULT_WORK);
  return {
    character: normalizeText(adapter?.character, modelCharacter),
    work: normalizeText(adapter?.work, modelWork),
  };
}

function buildModelProfiles(modelList = []) {
  return modelList.flatMap((model) =>
    (model.adapters ?? []).map((adapter) => {
      const context = resolveParticipantContext(model, adapter.path);
      return {
        modelName: model.name,
        adapterPath: adapter.path,
        character: context.character,
        work: context.work,
      };
    }),
  );
}

function buildCharacterOptions(modelList = []) {
  const characterSet = new Set();
  buildModelProfiles(modelList).forEach((profile) => {
    if (profile.character) {
      characterSet.add(profile.character);
    }
  });
  return characterSet.size > 0 ? Array.from(characterSet) : CHARACTER_OPTIONS;
}

function findProfileForCharacter(modelList = [], characterName = "") {
  const targetCharacter = normalizeText(characterName).toLowerCase();
  if (!targetCharacter) return null;

  return (
    buildModelProfiles(modelList).find(
      (profile) => profile.character.toLowerCase() === targetCharacter,
    ) ?? null
  );
}

function createMultiModelParticipant(index, modelList = []) {
  const modelProfiles = buildModelProfiles(modelList);
  const defaultProfile = modelProfiles[index % modelProfiles.length] ?? null;
  return {
    name: `Speaker ${index + 1}`,
    character: defaultProfile?.character || CHARACTER_OPTIONS[0],
    work: defaultProfile?.work || DEFAULT_WORK,
    model_name: defaultProfile?.modelName || "",
    adapter_path: defaultProfile?.adapterPath || "",
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

function normalizeVoiceOptions(payload) {
  const rawVoices = Array.isArray(payload?.voices)
    ? payload.voices
    : Array.isArray(payload)
      ? payload
      : [];

  return rawVoices
    .map((voice) => {
      if (!voice || typeof voice !== "object") return null;
      const name = typeof voice.name === "string" ? voice.name.trim() : "";
      if (!name) return null;
      return {
        name,
        voiceId:
          typeof voice.voice_id === "string" ? voice.voice_id.trim() : "",
      };
    })
    .filter(Boolean);
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
      message,
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
  const [activeTab, setActiveTab] = useState("single");
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
  const [isShakespeareStyleEnabled, setIsShakespeareStyleEnabled] =
    useState(false);
  const [multiModelConfig, setMultiModelConfig] = useState({
    defaultMaxTurns: MULTIMODEL_DEFAULT_MAX_TURNS,
    hardMaxTurns: MULTIMODEL_HARD_MAX_TURNS,
    minParticipants: MULTIMODEL_MIN_SPEAKERS,
    maxParticipants: MULTIMODEL_MAX_SPEAKERS,
  });
  const [multiDraft, setMultiDraft] = useState(
    "Debate whether action or patience better serves Denmark.",
  );
  const [multiConversationPrompt, setMultiConversationPrompt] = useState("");
  const [multiMaxTurns, setMultiMaxTurns] = useState(
    MULTIMODEL_DEFAULT_MAX_TURNS,
  );
  const [multiSpeakerCount, setMultiSpeakerCount] = useState(
    MULTIMODEL_MIN_SPEAKERS,
  );
  const [multiParticipants, setMultiParticipants] = useState(() =>
    Array.from({ length: MULTIMODEL_MIN_SPEAKERS }, (_, index) =>
      createMultiModelParticipant(index),
    ),
  );
  const [multiTurns, setMultiTurns] = useState([]);
  const [multiStatus, setMultiStatus] = useState(
    "Configure speakers to begin.",
  );
  const [multiError, setMultiError] = useState("");
  const [isMultiRunning, setIsMultiRunning] = useState(false);
  const [voiceOptions, setVoiceOptions] = useState([]);
  const [characterVoices, setCharacterVoices] = useState({});
  const [activityLog, setActivityLog] = useState([]);
  const [feedbackOpen, setFeedbackOpen] = useState(null);
  const [spans, setSpans] = useState({});
  const [votes, setVotes] = useState({});
  const [submittedFeedback, setSubmittedFeedback] = useState(new Set());
  const [pendingSpan, setPendingSpan] = useState(null);
  const [spanMenuPos, setSpanMenuPos] = useState({ x: 0, y: 0 });
  const bottomRef = useRef(null);
  const multiBottomRef = useRef(null);
  const activeAudioRef = useRef(null);
  const activeAudioUrlRef = useRef("");
  const pendingModelApplyCountRef = useRef(0);
  const multiStopRequestedRef = useRef(false);

  const modelDetails = useMemo(
    () => models.find((model) => model.name === selectedModel),
    [models, selectedModel],
  );
  const adapterOptions = useMemo(
    () => modelDetails?.adapters ?? [],
    [modelDetails],
  );
  const selectedAdapterDetails = useMemo(
    () =>
      adapterOptions.find((adapter) => adapter.path === selectedAdapter) ??
      null,
    [adapterOptions, selectedAdapter],
  );
  const availableCharacterOptions = useMemo(
    () => buildCharacterOptions(models),
    [models],
  );
  const visibleMultiParticipants = useMemo(
    () => multiParticipants.slice(0, multiSpeakerCount),
    [multiParticipants, multiSpeakerCount],
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

    const defaultProfiles = buildModelProfiles(models);
    setMultiParticipants((previous) =>
      previous.map((participant, index) => {
        const defaultProfile =
          defaultProfiles[index % defaultProfiles.length] ?? null;
        const currentModel = models.find(
          (model) => model.name === participant.model_name,
        );
        const defaultProfileModel = models.find(
          (model) => model.name === defaultProfile?.modelName,
        );
        const modelDetailsForParticipant =
          currentModel ?? defaultProfileModel ?? models[0];
        const adapterStillValid = modelDetailsForParticipant.adapters.some(
          (adapter) => adapter.path === participant.adapter_path,
        );
        const fallbackAdapterPath =
          !currentModel && defaultProfile?.adapterPath
            ? defaultProfile.adapterPath
            : resolveDefaultAdapterPath(modelDetailsForParticipant);
        const adapterPath = adapterStillValid
          ? participant.adapter_path
          : fallbackAdapterPath;
        const participantContext = resolveParticipantContext(
          modelDetailsForParticipant,
          adapterPath,
        );

        return {
          ...participant,
          model_name: modelDetailsForParticipant.name,
          adapter_path: adapterPath,
          character: participantContext.character,
          work: participantContext.work,
          name: participant.name || `Speaker ${index + 1}`,
        };
      }),
    );
  }, [models]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  useEffect(() => {
    if (activeTab === "multi") {
      multiBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [activeTab, multiConversationPrompt, multiTurns, isMultiRunning]);

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

  const handleTextSelect = (messageId) => {
    const selection = window.getSelection();
    const selected = selection?.toString().trim();
    if (!selected || !selection.rangeCount) return;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    setPendingSpan({ text: selected, messageId });
    setSpanMenuPos({
      x: rect.left,
      y: rect.top + window.scrollY - 45,
    });
  };

  const assignPolarity = (polarity) => {
    if (!pendingSpan) return;
    setSpans((prev) => ({
      ...prev,
      [pendingSpan.messageId]: [
        ...(prev[pendingSpan.messageId] ?? []),
        { text: pendingSpan.text, polarity },
      ],
    }));
    setPendingSpan(null);
    window.getSelection()?.removeAllRanges();
  };

  const handleVote = (messageId, vote) => {
    setVotes((prev) => ({ ...prev, [messageId]: vote }));
  };

  const handleFeedbackSubmit = async (message) => {
    try {
      await apiPostJson("/feedback", {
        conversation_id: message.conversationId,
        message_index: message.messageIndex,
        vote: votes[message.id] ?? "neutral",
        spans: spans[message.id] ?? [],
      });
      setSubmittedFeedback((prev) => new Set([...prev, message.id]));
      setFeedbackOpen(null);
      updateStatus("Feedback submitted.", "feedback");
    } catch (feedbackError) {
      reportError(feedbackError.message, "Feedback submission failed.");
    }
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
        `Refresh fallback used after /refresh_chat failed: ${refreshError.message}`,
      );
    }
    setMessages([]);
    setFeedbackOpen(null);
    setSpans({});
    setVotes({});
    setSubmittedFeedback(new Set());
    setPendingSpan(null);
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

  const fetchVoiceOptions = async () => {
    const payload = await apiGet("/voices");
    const nextVoices = normalizeVoiceOptions(payload);
    setVoiceOptions(nextVoices);
    return nextVoices;
  };

  const resolveVoiceForCharacter = (characterName) => {
    const stored = characterVoices[characterName];
    if (stored && voiceOptions.some((option) => option.name === stored)) {
      return stored;
    }
    return voiceOptions[0]?.name || DEFAULT_VOICE_OPTION;
  };

  const handleCharacterVoiceChange = (characterName, nextVoice) => {
    if (!characterName) return;
    setCharacterVoices((previous) => ({
      ...previous,
      [characterName]: nextVoice,
    }));
    updateStatus(
      `Voice for ${characterName} set to ${nextVoice}.`,
      "voice",
    );
  };

  const fetchMultiModelConfig = async () => {
    const payload = await apiGet("/multimodel/config");
    const config = normalizeMultiModelConfig(payload);
    setMultiModelConfig(config);
    setMultiMaxTurns(config.defaultMaxTurns);
    setMultiSpeakerCount((previousCount) =>
      Math.min(
        Math.max(previousCount, config.minParticipants),
        config.maxParticipants,
      ),
    );
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

  const applyCharacter = async (
    nextCharacter = character,
    showStatus = true,
  ) => {
    setError("");
    releaseActiveAudio();
    clearPlaybackState();
    await apiGet("/select_character", {
      character: nextCharacter,
      work: DEFAULT_WORK,
    });
    setMessages([]);
    setFeedbackOpen(null);
    setSpans({});
    setVotes({});
    setSubmittedFeedback(new Set());
    setPendingSpan(null);
    if (showStatus) {
      updateStatus(`Character set to ${nextCharacter}.`, "character");
    }
  };

  const applyModel = async (
    nextModel = selectedModel,
    nextAdapter = selectedAdapter,
    showStatus = true,
  ) => {
    setError("");
    if (!nextModel || !nextAdapter) {
      throw new Error("Select a valid model and adapter before loading.");
    }

    const activeModel = models.find((model) => model.name === nextModel);
    const activeAdapter = activeModel?.adapters.find(
      (adapter) => adapter.path === nextAdapter,
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
          "model",
        );
      }
    } finally {
      pendingModelApplyCountRef.current = Math.max(
        0,
        pendingModelApplyCountRef.current - 1,
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
      reportError(applyError.message, "Model apply failed."),
    );
  };

  const handleAdapterChange = (nextAdapter) => {
    setSelectedAdapter(nextAdapter);
    applyModel(selectedModel, nextAdapter).catch((applyError) =>
      reportError(applyError.message, "Model apply failed."),
    );
  };

  const handleCharacterChange = (nextCharacter) => {
    setCharacter(nextCharacter);
    applyCharacter(nextCharacter).catch((characterError) =>
      reportError(characterError.message, "Character update failed."),
    );
  };

  const handleStyleToggle = () => {
    setIsShakespeareStyleEnabled((isEnabled) => {
      const nextValue = !isEnabled;
      updateStatus(
        nextValue
          ? "Shakespeare dialogue polish enabled."
          : "Shakespeare dialogue polish disabled.",
        "style",
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
        `Turn limit must be between 1 and ${multiModelConfig.hardMaxTurns}.`,
      );
    }
    return wholeTurns;
  };

  const handleSaveMultiMaxTurns = async () => {
    setMultiError("");
    try {
      const savedConfig = await saveMultiModelConfig(parseMultiMaxTurns());
      setMultiStatus(
        `Default model conversation limit saved at ${savedConfig.defaultMaxTurns} turn(s).`,
      );
      recordActivity(
        "multimodel",
        `Default model conversation limit saved: ${savedConfig.defaultMaxTurns}.`,
      );
    } catch (configError) {
      setMultiError(configError.message);
      recordActivity("error", configError.message);
    }
  };

  const updateMultiParticipant = (index, updates) => {
    setMultiParticipants((previous) =>
      previous.map((participant, participantIndex) =>
        participantIndex === index
          ? { ...participant, ...updates }
          : participant,
      ),
    );
  };

  const handleMultiModelChange = (index, nextModelName) => {
    const nextModelDetails = models.find((model) => model.name === nextModelName);
    const nextAdapterPath = resolveDefaultAdapterPath(nextModelDetails);
    const participantContext = resolveParticipantContext(
      nextModelDetails,
      nextAdapterPath,
    );
    updateMultiParticipant(index, {
      model_name: nextModelName,
      adapter_path: nextAdapterPath,
      character: participantContext.character,
      work: participantContext.work,
    });
  };

  const handleMultiAdapterChange = (index, nextAdapterPath) => {
    const participant = multiParticipants[index];
    const participantModel = models.find(
      (model) => model.name === participant?.model_name,
    );
    const participantContext = resolveParticipantContext(
      participantModel,
      nextAdapterPath,
    );
    updateMultiParticipant(index, {
      adapter_path: nextAdapterPath,
      character: participantContext.character,
      work: participantContext.work,
    });
  };

  const handleMultiCharacterChange = (index, nextCharacter) => {
    const matchedProfile = findProfileForCharacter(models, nextCharacter);
    if (!matchedProfile) {
      updateMultiParticipant(index, {
        character: nextCharacter,
        work: DEFAULT_WORK,
      });
      return;
    }

    updateMultiParticipant(index, {
      character: matchedProfile.character,
      work: matchedProfile.work,
      model_name: matchedProfile.modelName,
      adapter_path: matchedProfile.adapterPath,
    });
  };

  const handleMultiSpeakerCountChange = (nextCountValue) => {
    const parsedCount = Number(nextCountValue);
    const requestedCount = Number.isFinite(parsedCount)
      ? parsedCount
      : multiModelConfig.minParticipants;
    const boundedCount = Math.min(
      Math.max(requestedCount, multiModelConfig.minParticipants),
      multiModelConfig.maxParticipants,
    );
    setMultiSpeakerCount(boundedCount);
    setMultiParticipants((previous) => {
      const nextParticipants = [...previous];
      while (nextParticipants.length < boundedCount) {
        nextParticipants.push(
          createMultiModelParticipant(nextParticipants.length, models),
        );
      }
      return nextParticipants;
    });
  };

  const buildMultiStartPayload = (initialPromptText) => {
    const initialPrompt = initialPromptText.trim();
    if (!initialPrompt) {
      throw new Error("Enter a prompt for the model conversation.");
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
        !participant.adapter_path,
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

  const handleMultiSend = async (event) => {
    event.preventDefault();
    if (isMultiRunning || isSending || isApplyingModel) {
      return;
    }

    const initialPrompt = multiDraft.trim();
    if (!initialPrompt) {
      return;
    }

    let startPayload;
    try {
      startPayload = buildMultiStartPayload(initialPrompt);
    } catch (validationError) {
      setMultiError(validationError.message);
      return;
    }

    setMultiError("");
    setMultiConversationPrompt(initialPrompt);
    setMultiDraft("");
    setMultiTurns([]);
    setIsMultiRunning(true);
    multiStopRequestedRef.current = false;
    releaseActiveAudio();
    clearPlaybackState();
    setMultiStatus("Starting model conversation...");
    recordActivity("multimodel", `Prompt sent: ${initialPrompt}`);

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
            `${session.last_turn.speaker_name} added turn ${session.last_turn.turn_number}.`,
          );
        }
      }

      if (multiStopRequestedRef.current || session?.status === "stopped") {
        setMultiStatus("Model conversation stopped.");
      } else {
        setMultiStatus(
          `Model conversation complete at ${session?.turn_count || 0} turn(s).`,
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
              `Waiting for backend to start... retry ${nextAttempt}/${STARTUP_RETRY_ATTEMPTS}.`,
            );
          },
        });
        if (cancelled) return;
        recordActivity("character", "Default character context applied.");

        try {
          const config = await fetchMultiModelConfig();
          recordActivity(
            "multimodel",
            `Model conversation limit set to ${config.defaultMaxTurns} turn(s).`,
          );
        } catch (configError) {
          recordActivity(
            "multimodel",
            `Using default model conversation limit after config failed: ${configError.message}`,
          );
        }

        try {
          const fetchedVoices = await fetchVoiceOptions();
          recordActivity(
            "voice",
            `Loaded ${fetchedVoices.length} voice option(s).`,
          );
        } catch (voicesError) {
          recordActivity(
            "voice",
            `Voice options unavailable: ${voicesError.message}`,
          );
        }

        const loadedModels = await retryStartupAction(
          () => fetchModels(false),
          {
            isCancelled: () => cancelled,
            onRetry: (nextAttempt) => {
              setStatus(
                `Backend reached. Loading models... retry ${nextAttempt}/${STARTUP_RETRY_ATTEMPTS}.`,
              );
            },
          },
        );
        if (cancelled) return;
        recordActivity(
          "models",
          `Discovered ${loadedModels.length} loadable model option(s).`,
        );

        const firstModel = loadedModels[0];
        const firstAdapter = resolveDefaultAdapterPath(firstModel);
        if (firstModel?.name && firstAdapter) {
          setSelectedModel(firstModel.name);
          setSelectedAdapter(firstAdapter);
          await applyModel(firstModel.name, firstAdapter, false);
          recordActivity(
            "model",
            `Default model loaded: ${firstModel.name} using ${firstAdapter}.`,
          );
        }

        if (!cancelled) {
          updateStatus(
            firstModel?.name && firstAdapter
              ? "Thy chatbot is ready."
              : "No loadable models are currently available.",
            "startup",
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
          conversationId: payload?.conversation_id ?? "",
          messageIndex: payload?.message_index ?? 0,
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
      const audioBlob = await apiPostBlob("/tts", {
        text,
        character,
        voice: resolveVoiceForCharacter(character),
      });
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

  const isSingleTab = activeTab === "single";
  const isMultiTab = activeTab === "multi";
  const singleInputDisabled = isSending || isApplyingModel || isMultiRunning;
  const multiInputDisabled =
    isMultiRunning || isSending || isApplyingModel || models.length === 0;
  const displayedStatus = isMultiTab ? multiStatus : status;
  const displayedError = isMultiTab ? multiError : error;

  return (
    <div className="app-shell mx-auto flex min-h-screen w-full flex-col px-4 py-8 md:px-8">
      <header className="folio-header rounded-2xl border-2 border-maroon bg-white px-5 py-6 shadow-[0_10px_30px_rgba(165,46,48,0.16)]">
        <h1 className="folio-title break-words text-center font-hamlet text-[clamp(1.6rem,5vw,3.4rem)] leading-tight text-maroon">
          Shakespearean Character Language Models
        </h1>
      </header>

      <nav
        className="mode-tabs mt-4 inline-flex w-full rounded-xl border border-maroon/25 bg-white p-1 shadow-[0_6px_18px_rgba(69,20,21,0.08)] sm:w-auto"
        role="tablist"
        aria-label="Dialogue modes"
      >
        <button
          className={`mode-tab flex-1 rounded-lg px-4 py-2 text-sm font-semibold transition sm:flex-none ${
            isSingleTab
              ? "mode-tab-active bg-maroon text-white"
              : "text-maroon hover:bg-gold/20"
          }`}
          type="button"
          role="tab"
          aria-selected={isSingleTab}
          onClick={() => setActiveTab("single")}
        >
          Single Character Dialogue
        </button>
        <button
          className={`mode-tab flex-1 rounded-lg px-4 py-2 text-sm font-semibold transition sm:flex-none ${
            isMultiTab
              ? "mode-tab-active bg-maroon text-white"
              : "text-maroon hover:bg-gold/20"
          }`}
          type="button"
          role="tab"
          aria-selected={isMultiTab}
          onClick={() => setActiveTab("multi")}
        >
          Multi-Model Dialogue
        </button>
      </nav>

      <section className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <article className="rounded-2xl border border-maroon/20 bg-white px-4 py-4 shadow-[0_6px_18px_rgba(69,20,21,0.08)]">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-maroon/60">
            Current Status
          </p>
          <p className="mt-2 text-lg text-maroon">{displayedStatus}</p>
          {displayedError && (
            <p className="mt-3 rounded-xl border border-maroon/20 bg-maroon/5 px-3 py-2 text-sm text-maroon">
              {displayedError}
            </p>
          )}
        </article>

        <article className="rounded-2xl border border-maroon/20 bg-white px-4 py-4 shadow-[0_6px_18px_rgba(69,20,21,0.08)]">
          <div className="flex items-center justify-between gap-3">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-maroon/60">
              Activity Log
            </p>
            <span className="text-xs text-maroon/60">
              {activityLog.length} recent event
              {activityLog.length === 1 ? "" : "s"}
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
                  <span className="text-xs text-maroon/60">
                    {entry.timestamp}
                  </span>
                </div>
                <p className="mt-1 leading-snug">{entry.detail}</p>
              </div>
            ))}
          </div>
        </article>
      </section>

      {isSingleTab && (
        <>
          <section className="mt-6">
            <details
              className="settings-panel rounded-2xl border border-maroon/25 bg-white p-4"
              open
            >
              <summary className="cursor-pointer font-semibold text-maroon">
                Single-character settings
              </summary>

              <div className="mt-3 grid gap-3 md:grid-cols-2 lg:grid-cols-4">
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
                    {modelDetails?.description ||
                      "No model description available."}
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
                    onChange={(event) =>
                      handleCharacterChange(event.target.value)
                    }
                    disabled={isMultiRunning}
                  >
                    {availableCharacterOptions.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-maroon">
                    Voice
                  </label>
                  <select
                    className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                    value={resolveVoiceForCharacter(character)}
                    onChange={(event) =>
                      handleCharacterVoiceChange(character, event.target.value)
                    }
                    disabled={isMultiRunning || voiceOptions.length === 0}
                  >
                    {voiceOptions.map((option) => (
                      <option key={option.name} value={option.name}>
                        {option.name}
                      </option>
                    ))}
                    {voiceOptions.length === 0 && (
                      <option>No voices available</option>
                    )}
                  </select>
                  <p className="mt-1 min-h-10 text-sm text-maroon/75">
                    {voiceOptions.length === 0
                      ? "Voice list unavailable. Check the TTS backend."
                      : `Voice played for ${character}.`}
                  </p>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                <p className="text-sm text-maroon/80">
                  Selections apply automatically.
                </p>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    className={`manuscript-button rounded-lg border px-3 py-2 text-sm font-semibold ${
                      isShakespeareStyleEnabled
                        ? "seal-button border-maroon bg-maroon text-white"
                        : "border-maroon bg-white text-maroon"
                    }`}
                    onClick={handleStyleToggle}
                    type="button"
                    aria-pressed={isShakespeareStyleEnabled}
                    disabled={isMultiRunning}
                  >
                    Shakespeare Style:{" "}
                    {isShakespeareStyleEnabled ? "On" : "Off"}
                  </button>
                  <button
                    className="manuscript-button rounded-lg border border-maroon bg-white px-3 py-2 text-sm font-semibold text-maroon"
                    onClick={() =>
                      refreshServerChat().catch((refreshError) =>
                        reportError(refreshError.message, "Chat reset failed."),
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

          <section className="dialogue-folio mt-6 flex flex-1 flex-col rounded-2xl border-2 border-maroon bg-parchment p-4 shadow-[0_8px_24px_rgba(165,46,48,0.12)]">
            <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="dialogue-title text-xl font-semibold text-maroon">
                  Single Character Dialogue
                </h2>
                <p className="status-copy mt-1 text-sm text-maroon/75">
                  {status}
                </p>
              </div>
              {error && (
                <p className="notice rounded-xl border border-maroon/20 bg-white px-3 py-2 text-sm text-maroon">
                  {error}
                </p>
              )}
            </div>

            <div className="message-scroll h-[420px] overflow-y-auto pr-2">
              {messages.length === 0 && (
                <p className="empty-state pt-10 text-center text-lg text-maroon/75">
                  Speak to {character} to begin the conversation.
                </p>
              )}

              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`message-row mb-3 flex w-fit max-w-[96%] items-start gap-2 ${
                    message.role === "user"
                      ? "ml-auto flex-row-reverse"
                      : "mr-auto"
                  }`}
                >
                  <MessageAvatar
                    type={message.role === "user" ? "user" : "assistant"}
                  />

                  <article
                    className={`chat-bubble max-w-[92%] rounded-xl border px-4 py-3 ${
                      message.role === "user"
                        ? "user-bubble border-maroon bg-maroon text-white"
                        : "assistant-bubble border-gold bg-white text-maroon"
                    }`}
                  >
                    <p
                      className="whitespace-pre-wrap text-lg leading-relaxed"
                      onMouseUp={
                        message.role === "assistant"
                          ? () => handleTextSelect(message.id)
                          : undefined
                      }
                      style={{
                        userSelect:
                          message.role === "assistant" ? "text" : "auto",
                      }}
                    >
                      {message.content}
                    </p>
                    {message.role === "assistant" && (
                      <div className="assistant-actions mt-2 flex flex-wrap gap-2">
                        <button
                          className="rounded-md border border-maroon px-2 py-1 text-sm font-medium text-maroon hover:bg-gold disabled:cursor-not-allowed disabled:opacity-60"
                          onClick={() =>
                            handleSpeak(message.id, message.content)
                          }
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

                        {!submittedFeedback.has(message.id) ? (
                          <button
                            className="rounded-md border border-maroon px-2 py-1 text-sm font-medium text-maroon hover:bg-gold"
                            onClick={() =>
                              setFeedbackOpen(
                                feedbackOpen === message.id ? null : message.id,
                              )
                            }
                            type="button"
                          >
                            Rate Response
                          </button>
                        ) : (
                          <span className="px-2 py-1 text-sm text-maroon/60">
                            ✓ Feedback submitted
                          </span>
                        )}

                        {feedbackOpen === message.id && (
                          <div className="mt-2 w-full rounded-xl border border-maroon/30 bg-parchment p-3">
                            <div className="mb-2 flex justify-end">
                              <button
                                onClick={() => setFeedbackOpen(null)}
                                className="text-xs text-maroon/60 hover:text-maroon"
                                type="button"
                              >
                                ▲ Close
                              </button>
                            </div>

                            {(spans[message.id] ?? []).length > 0 && (
                              <div className="mb-2 flex flex-wrap gap-1 text-sm">
                                {(spans[message.id] ?? []).map((span, idx) => (
                                  <span
                                    key={`${span.text}-${idx}`}
                                    onClick={() =>
                                      setSpans((prev) => ({
                                        ...prev,
                                        [message.id]: prev[message.id].filter(
                                          (_, spanIndex) => spanIndex !== idx,
                                        ),
                                      }))
                                    }
                                    className={`cursor-pointer rounded px-2 py-0.5 text-xs hover:opacity-60 ${
                                      span.polarity === "good"
                                        ? "bg-green-100 text-green-800"
                                        : "bg-red-100 text-red-800"
                                    }`}
                                  >
                                    {span.polarity === "good" ? "✓" : "✗"} "
                                    {span.text}" ✕
                                  </span>
                                ))}
                              </div>
                            )}

                            <p className="mb-2 text-xs text-maroon/60">
                              Highlight text above then mark it good or bad.
                              Also upvote or downvote the full response.
                            </p>

                            <div className="flex flex-wrap gap-2">
                              <button
                                className={`rounded-md border px-3 py-1 text-sm font-medium ${
                                  votes[message.id] === "up"
                                    ? "border-green-600 bg-green-100 text-green-800"
                                    : "border-maroon text-maroon hover:bg-gold"
                                }`}
                                onClick={() =>
                                  handleVote(
                                    message.id,
                                    votes[message.id] === "up" ? null : "up",
                                  )
                                }
                                type="button"
                              >
                                ▲ Upvote
                              </button>
                              <button
                                className={`rounded-md border px-3 py-1 text-sm font-medium ${
                                  votes[message.id] === "down"
                                    ? "border-red-600 bg-red-100 text-red-800"
                                    : "border-maroon text-maroon hover:bg-gold"
                                }`}
                                onClick={() =>
                                  handleVote(
                                    message.id,
                                    votes[message.id] === "down"
                                      ? null
                                      : "down",
                                  )
                                }
                                type="button"
                              >
                                ▼ Downvote
                              </button>
                              <button
                                className="rounded-md border border-maroon bg-maroon px-3 py-1 text-sm font-medium text-white hover:bg-maroon/80"
                                onClick={() => handleFeedbackSubmit(message)}
                                type="button"
                              >
                                Submit
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </article>
                </div>
              ))}

              {isSending && (
                <div className="message-row mb-3 flex max-w-[96%] items-start gap-2">
                  <MessageAvatar />

                  <article className="typing-indicator chat-bubble assistant-bubble max-w-[92%] rounded-xl border border-gold bg-white px-4 py-3 text-maroon">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-semibold text-maroon/75">
                        {character} is drafting
                      </span>
                      <div
                        className="flex items-center gap-1.5"
                        aria-hidden="true"
                      >
                        <span className="typing-dot" />
                        <span className="typing-dot typing-dot-delay-1" />
                        <span className="typing-dot typing-dot-delay-2" />
                      </div>
                    </div>
                  </article>
                </div>
              )}

              {pendingSpan && (
                <div
                  style={{
                    position: "fixed",
                    top: spanMenuPos.y,
                    left: spanMenuPos.x,
                    zIndex: 4000,
                  }}
                  className="flex gap-1 rounded-xl border border-maroon bg-white p-1 shadow-lg"
                >
                  <button
                    className="rounded-md bg-green-100 px-3 py-1 text-sm font-medium text-green-800 hover:bg-green-200"
                    onClick={() => assignPolarity("good")}
                    type="button"
                  >
                    👍 Good
                  </button>
                  <button
                    className="rounded-md bg-red-100 px-3 py-1 text-sm font-medium text-red-800 hover:bg-red-200"
                    onClick={() => assignPolarity("bad")}
                    type="button"
                  >
                    👎 Bad
                  </button>
                  <button
                    className="rounded-md bg-gray-100 px-3 py-1 text-sm text-maroon hover:bg-gray-200"
                    onClick={() => setPendingSpan(null)}
                    type="button"
                  >
                    ✕
                  </button>
                </div>
              )}

              <div ref={bottomRef} />
            </div>

            <form className="composer-form mt-4 flex gap-2" onSubmit={handleSend}>
              <input
                className="manuscript-input flex-1 rounded-xl border border-maroon/35 bg-white px-4 py-3 text-lg text-maroon placeholder:text-maroon/50 focus:border-maroon focus:outline-none"
                placeholder="What sayest thou?"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                disabled={singleInputDisabled}
              />
              <button
                type="submit"
                disabled={singleInputDisabled}
                className="send-quill-btn quill-send-button inline-flex h-12 min-w-12 items-center justify-center rounded-lg border-2 border-gold bg-white px-3 shadow-sm transition hover:bg-gold/20 disabled:cursor-not-allowed disabled:opacity-60"
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
        </>
      )}

      {isMultiTab && (
        <section className="dialogue-folio mt-6 flex flex-1 flex-col rounded-2xl border-2 border-maroon bg-parchment p-4 shadow-[0_8px_24px_rgba(165,46,48,0.12)]">
          <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="dialogue-title text-xl font-semibold text-maroon">
                Multi-Model Dialogue
              </h2>
              <p className="status-copy mt-1 text-sm text-maroon/75">
                {multiStatus}
              </p>
            </div>
            <button
              className="manuscript-button rounded-lg border border-maroon bg-white px-3 py-2 text-sm font-semibold text-maroon disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStopMultiConversation}
              type="button"
              disabled={!isMultiRunning}
            >
              Stop
            </button>
          </div>

          <details className="settings-panel mb-4 rounded-xl border border-maroon/25 bg-white p-4">
            <summary className="cursor-pointer font-semibold text-maroon">
              Multi-model settings
            </summary>

            <div className="mt-3 grid gap-3 md:grid-cols-[minmax(0,1fr)_12rem_12rem]">
              <div>
                <span className="text-sm font-medium text-maroon">
                  Shakespeare Style
                </span>
                <button
                  className={`manuscript-button mt-1 w-full rounded-lg border px-3 py-2 text-sm font-semibold ${
                    isShakespeareStyleEnabled
                      ? "seal-button border-maroon bg-maroon text-white"
                      : "border-maroon bg-white text-maroon"
                  }`}
                  onClick={handleStyleToggle}
                  type="button"
                  aria-pressed={isShakespeareStyleEnabled}
                  disabled={isMultiRunning}
                >
                  {isShakespeareStyleEnabled ? "On" : "Off"}
                </button>
              </div>

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
                    (_, index) => multiModelConfig.minParticipants + index,
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
                  className="manuscript-button mt-2 w-full rounded-lg border border-maroon bg-white px-3 py-2 text-sm font-semibold text-maroon disabled:cursor-not-allowed disabled:opacity-60"
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
                  (model) => model.name === participant.model_name,
                );
                const participantAdapters = participantModel?.adapters ?? [];
                const participantAdapterDetails =
                  participantAdapters.find(
                    (adapter) => adapter.path === participant.adapter_path,
                  ) ?? null;

                return (
                  <article
                    key={`${index}-${participant.name}`}
                    className="settings-panel rounded-2xl border border-maroon/25 bg-white p-4"
                  >
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <h3 className="text-base font-semibold text-maroon">
                        {participant.name || `Speaker ${index + 1}`}
                      </h3>
                      <span className="rounded-md border border-gold bg-parchment px-2 py-1 text-xs font-semibold text-maroon/70">
                        Speaker {index + 1}
                      </span>
                    </div>

                    <div className="grid gap-3 md:grid-cols-2">
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
                          {models.length === 0 && (
                            <option>No models available</option>
                          )}
                        </select>
                        <p className="mt-1 min-h-10 text-sm text-maroon/75">
                          {participantModel?.description ||
                            "No model description available."}
                        </p>
                      </label>

                      <label className="block">
                        <span className="text-sm font-medium text-maroon">
                          Adapter
                        </span>
                        <select
                          className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                          value={participant.adapter_path}
                          onChange={(event) =>
                            handleMultiAdapterChange(index, event.target.value)
                          }
                          disabled={
                            isMultiRunning || participantAdapters.length === 0
                          }
                        >
                          {participantAdapters.map((adapter) => (
                            <option key={adapter.path} value={adapter.path}>
                              {adapter.name}
                            </option>
                          ))}
                          {participantAdapters.length === 0 && (
                            <option>No adapter</option>
                          )}
                        </select>
                        <p className="mt-1 min-h-10 text-sm text-maroon/75">
                          {participantAdapterDetails?.description ||
                            "No adapter description available."}
                        </p>
                      </label>
                    </div>

                    <div className="mt-3 grid gap-2 sm:grid-cols-2">
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
                        <select
                          className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                          value={participant.character}
                          onChange={(event) =>
                            handleMultiCharacterChange(index, event.target.value)
                          }
                          disabled={isMultiRunning}
                        >
                          {availableCharacterOptions.map((name) => (
                            <option key={name} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>

                    <div className="mt-3">
                      <label className="block">
                        <span className="text-sm font-medium text-maroon">
                          Voice
                        </span>
                        <select
                          className="mt-1 w-full rounded-lg border border-maroon/30 bg-white px-3 py-2 text-base text-maroon"
                          value={resolveVoiceForCharacter(participant.character)}
                          onChange={(event) =>
                            handleCharacterVoiceChange(
                              participant.character,
                              event.target.value,
                            )
                          }
                          disabled={
                            isMultiRunning || voiceOptions.length === 0
                          }
                        >
                          {voiceOptions.map((option) => (
                            <option key={option.name} value={option.name}>
                              {option.name}
                            </option>
                          ))}
                          {voiceOptions.length === 0 && (
                            <option>No voices available</option>
                          )}
                        </select>
                      </label>
                    </div>

                    <div className="mt-3 rounded-lg border border-gold/60 bg-parchment px-3 py-2 text-sm text-maroon/75">
                      Work: {participant.work || DEFAULT_WORK}
                    </div>
                  </article>
                );
              })}
            </div>
          </details>

          {multiError && (
            <p className="notice mb-3 rounded-xl border border-maroon/20 bg-white px-3 py-2 text-sm text-maroon">
              {multiError}
            </p>
          )}

          <div className="message-scroll h-[420px] overflow-y-auto pr-2">
            {!multiConversationPrompt && multiTurns.length === 0 && (
              <p className="empty-state pt-10 text-center text-lg text-maroon/75">
                Send a prompt to start a model-to-model dialogue.
              </p>
            )}

            {multiConversationPrompt && (
              <div className="message-row mb-3 ml-auto flex max-w-[96%] flex-row-reverse items-start justify-end gap-2">
                <MessageAvatar type="user" />
                <article className="chat-bubble user-bubble max-w-[92%] rounded-xl border border-maroon bg-maroon px-4 py-3 text-white">
                  <p className="whitespace-pre-wrap text-lg leading-relaxed">
                    {multiConversationPrompt}
                  </p>
                </article>
              </div>
            )}

            {multiTurns.map((turn) => {
              const alignRight = Number(turn.speaker_index) % 2 === 1;
              return (
                <div
                  key={`${turn.turn_number}-${turn.speaker_name}`}
                  className={`message-row mb-3 flex max-w-[96%] items-start gap-2 ${
                    alignRight
                      ? "ml-auto flex-row-reverse justify-end"
                      : "mr-auto"
                  }`}
                >
                  <MessageAvatar />

                  <article className="chat-bubble assistant-bubble max-w-[92%] rounded-xl border border-gold bg-white px-4 py-3 text-maroon">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-sm font-semibold text-maroon/75">
                        {turn.speaker_name} as {turn.character}
                      </p>
                      <span className="text-xs text-maroon/60">
                        Turn {turn.turn_number}
                      </span>
                    </div>
                    <p className="mt-2 whitespace-pre-wrap text-lg leading-relaxed">
                      {turn.content}
                    </p>
                  </article>
                </div>
              );
            })}

            {isMultiRunning && (
              <div className="message-row mb-3 flex max-w-[96%] items-start gap-2">
                <MessageAvatar />

                <article className="typing-indicator chat-bubble assistant-bubble max-w-[92%] rounded-xl border border-gold bg-white px-4 py-3 text-maroon">
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-semibold text-maroon/75">
                      {multiStatus}
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

            <div ref={multiBottomRef} />
          </div>

          <form className="composer-form mt-4 flex gap-2" onSubmit={handleMultiSend}>
            <input
              className="manuscript-input flex-1 rounded-xl border border-maroon/35 bg-white px-4 py-3 text-lg text-maroon placeholder:text-maroon/50 focus:border-maroon focus:outline-none"
              placeholder={
                models.length === 0
                  ? "No models available"
                  : "Start a model-to-model exchange"
              }
              value={multiDraft}
              onChange={(event) => setMultiDraft(event.target.value)}
              disabled={multiInputDisabled}
            />
            <button
              type="submit"
              disabled={multiInputDisabled || multiDraft.trim().length === 0}
              className="send-quill-btn quill-send-button inline-flex h-12 min-w-12 items-center justify-center rounded-lg border-2 border-gold bg-white px-3 shadow-sm transition hover:bg-gold/20 disabled:cursor-not-allowed disabled:opacity-60"
              aria-label="Send multimodel prompt"
            >
              {isMultiRunning ? (
                <span className="inline-flex items-center gap-2 text-sm font-semibold text-maroon">
                  <span className="loading-ripple" aria-hidden="true" />
                  Running
                </span>
              ) : isApplyingModel ? (
                <span className="inline-flex items-center gap-2 text-sm font-semibold text-maroon">
                  Applying
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
      )}

      <footer className="folio-footer mt-3 min-h-6 text-sm text-maroon/70">
        <p>Latest event: {activityLog[0]?.detail || displayedStatus}</p>
      </footer>
    </div>
  );
}
