use std::net::SocketAddr;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};

use qwen3_tts::{
    device_info, parse_device, AudioBuffer, Language, ModelType, Qwen3TTS, SynthesisOptions,
    VoiceClonePrompt,
};

const FRAME_RATE_HZ: f64 = 12.5;

#[derive(Clone)]
struct AppState {
    base: Arc<Mutex<Qwen3TTS>>,
    voice_design: Arc<Mutex<Qwen3TTS>>,
    base_model_dir: String,
    _voice_design_model_dir: String,
    ffmpeg_path: String,
    auto_ref_text: bool,
    transcribe_path: String,
    whisper_model: String,
    whisper_language: String,
    include_ref_audio: bool,
    voice_design_min_duration: Option<f64>,
    semaphore: Arc<Semaphore>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            error: self.message,
        });
        (self.status, body).into_response()
    }
}

#[derive(Debug, Deserialize)]
struct SynthesisOptionsRequest {
    max_frames: Option<usize>,
    duration_seconds: Option<f64>,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    repetition_penalty: Option<f64>,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct VoiceCloneRequest {
    text: String,
    prompt: PromptPayload,
    language: Option<String>,
    options: Option<SynthesisOptionsRequest>,
}

#[derive(Debug, Deserialize)]
struct VoiceDesignRequest {
    text: String,
    instruct: String,
    language: Option<String>,
    options: Option<SynthesisOptionsRequest>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PromptPayload {
    speaker_embedding: Vec<f32>,
    ref_codes: Option<Vec<Vec<u32>>>,
    ref_text_ids: Option<Vec<u32>>,
    #[serde(default)]
    ref_text: Option<String>,
    #[serde(default)]
    ref_audio_b64: Option<String>,
}

#[derive(Debug, Serialize)]
struct PromptResponse {
    prompt: PromptPayload,
    model_type: String,
    model_dir: String,
}

fn options_from_request(req: Option<SynthesisOptionsRequest>) -> Result<SynthesisOptions, ApiError> {
    let mut options = SynthesisOptions::default();
    if let Some(req) = req {
        if let Some(duration) = req.duration_seconds {
            if duration <= 0.0 {
                return Err(ApiError::bad_request("duration_seconds must be > 0"));
            }
            options.max_length = (duration * FRAME_RATE_HZ) as usize;
            if options.max_length == 0 {
                options.max_length = 1;
            }
        } else if let Some(frames) = req.max_frames {
            if frames == 0 {
                return Err(ApiError::bad_request("max_frames must be > 0"));
            }
            options.max_length = frames;
        }

        if let Some(v) = req.temperature {
            options.temperature = v;
        }
        if let Some(v) = req.top_k {
            options.top_k = v;
        }
        if let Some(v) = req.top_p {
            options.top_p = v;
        }
        if let Some(v) = req.repetition_penalty {
            options.repetition_penalty = v;
        }
        if let Some(v) = req.seed {
            options.seed = Some(v);
        }
    }
    Ok(options)
}

fn parse_language(lang: Option<String>) -> Result<Language, ApiError> {
    let lang = lang.unwrap_or_else(|| "english".to_string());
    lang.parse()
        .map_err(|e| ApiError::bad_request(format!("invalid language: {e}")))
}

fn parse_env_bool(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false)
}

fn apply_min_duration(options: &mut SynthesisOptions, min_duration: Option<f64>) {
    if let Some(min_seconds) = min_duration {
        if min_seconds > 0.0 {
            let min_frames = (min_seconds * FRAME_RATE_HZ) as usize;
            if min_frames > 0 && options.max_length < min_frames {
                options.max_length = min_frames;
            }
        }
    }
}

fn transcribe_ref_text(
    wav_path: &Path,
    transcribe_path: &str,
    model: &str,
    language: &str,
) -> Result<String, ApiError> {
    let mut cmd = if transcribe_path.ends_with(".py") {
        let mut cmd = Command::new("python3");
        cmd.arg(transcribe_path);
        cmd
    } else {
        Command::new(transcribe_path)
    };

    let mut cmd = cmd.arg("--model").arg(model);
    if !language.is_empty() && language != "auto" {
        cmd = cmd.arg("--language").arg(language);
    }

    let output = cmd
        .arg("--json")
        .arg(wav_path)
        .output()
        .map_err(|e| ApiError::internal(format!("failed to run transcribe: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ApiError::internal(format!(
            "transcribe failed: {stderr}"
        )));
    }

    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| ApiError::internal(format!("transcribe JSON parse error: {e}")))?;
    let text = value
        .get(0)
        .and_then(|entry| entry.get("transcription"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if text.is_empty() {
        return Err(ApiError::internal(
            "transcribe returned empty transcription",
        ));
    }

    Ok(text)
}

fn prompt_to_payload(prompt: &VoiceClonePrompt) -> Result<PromptPayload, ApiError> {
    let speaker_embedding = prompt
        .speaker_embedding
        .to_vec1::<f32>()
        .map_err(|e| ApiError::internal(format!("failed to read speaker embedding: {e}")))?;

    let ref_codes = if let Some(codes) = &prompt.ref_codes {
        let codes = codes
            .contiguous()
            .map_err(|e| ApiError::internal(format!("ref_codes contiguous error: {e}")))?;
        let (n_frames, n_codebooks) = codes
            .dims2()
            .map_err(|e| ApiError::internal(format!("invalid ref_codes shape: {e}")))?;
        if n_codebooks != 16 {
            return Err(ApiError::bad_request(format!(
                "ref_codes must have 16 codebooks, got {}",
                n_codebooks
            )));
        }
        let codes_u32 = codes
            .to_dtype(DType::U32)
            .map_err(|e| ApiError::internal(format!("failed to convert ref_codes: {e}")))?;
        let flat = codes_u32
            .flatten_all()
            .map_err(|e| ApiError::internal(format!("ref_codes flatten error: {e}")))?
            .to_vec1::<u32>()
            .map_err(|e| ApiError::internal(format!("ref_codes flat read error: {e}")))?;
        if flat.len() != n_frames * n_codebooks {
            return Err(ApiError::internal(format!(
                "ref_codes size mismatch: got {}, expected {}",
                flat.len(),
                n_frames * n_codebooks
            )));
        }
        let mut frames = Vec::with_capacity(n_frames);
        for idx in 0..n_frames {
            let start = idx * n_codebooks;
            frames.push(flat[start..start + n_codebooks].to_vec());
        }
        Some(frames)
    } else {
        None
    };

    Ok(PromptPayload {
        speaker_embedding,
        ref_codes,
        ref_text_ids: prompt.ref_text_ids.clone(),
        ref_text: None,
        ref_audio_b64: None,
    })
}

fn payload_to_prompt(
    payload: PromptPayload,
    device: &candle_core::Device,
) -> Result<VoiceClonePrompt, ApiError> {
    let speaker_embedding = Tensor::new(payload.speaker_embedding.as_slice(), device)
        .map_err(|e| ApiError::internal(format!("failed to build speaker embedding: {e}")))?;

    let ref_codes = if let Some(frames) = payload.ref_codes {
        if frames.is_empty() {
            None
        } else {
            let frame_count = frames.len();
            let frame_len = frames[0].len();
            if frame_len == 0 {
                return Err(ApiError::bad_request("ref_codes rows must be non-empty"));
            }
            if frame_len != 16 {
                return Err(ApiError::bad_request(format!(
                    "ref_codes must have 16 codebooks, got {}",
                    frame_len
                )));
            }
            for (idx, row) in frames.iter().enumerate() {
                if row.len() != frame_len {
                    return Err(ApiError::bad_request(format!(
                        "ref_codes row {} has length {}, expected {}",
                        idx,
                        row.len(),
                        frame_len
                    )));
                }
            }
            let mut flat = Vec::with_capacity(frame_count * frame_len);
            for row in frames.iter() {
                flat.extend_from_slice(row);
            }
            let tensor = Tensor::from_vec(flat, (frame_count, frame_len), device)
                .map_err(|e| ApiError::internal(format!("failed to build ref_codes: {e}")))?;
            Some(tensor)
        }
    } else {
        None
    };

    Ok(VoiceClonePrompt {
        speaker_embedding,
        ref_codes,
        ref_text_ids: payload.ref_text_ids,
    })
}

fn audio_to_mp3(audio: &AudioBuffer, ffmpeg_path: &str) -> Result<Vec<u8>, ApiError> {
    let wav_file = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|e| ApiError::internal(format!("failed to create temp wav: {e}")))?;
    audio
        .save(wav_file.path())
        .map_err(|e| ApiError::internal(format!("failed to save wav: {e}")))?;

    let mp3_file = tempfile::Builder::new()
        .suffix(".mp3")
        .tempfile()
        .map_err(|e| ApiError::internal(format!("failed to create temp mp3: {e}")))?;

    let output = Command::new(ffmpeg_path)
        .arg("-y")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(wav_file.path())
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("24000")
        .arg("-codec:a")
        .arg("libmp3lame")
        .arg(mp3_file.path())
        .output()
        .map_err(|e| ApiError::internal(format!("failed to run ffmpeg: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ApiError::internal(format!("ffmpeg failed: {stderr}")));
    }

    std::fs::read(mp3_file.path())
        .map_err(|e| ApiError::internal(format!("failed to read mp3: {e}")))
}

async fn create_prompt(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<PromptResponse>, ApiError> {
    let mut ref_audio: Option<Vec<u8>> = None;
    let mut ref_text: Option<String> = None;
    let mut x_vector_only = false;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request(format!("invalid multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "ref_audio" => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("ref_audio read error: {e}")))?;
                ref_audio = Some(bytes.to_vec());
            }
            "ref_text" => {
                ref_text = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::bad_request(format!("ref_text read error: {e}")))?,
                );
            }
            "x_vector_only" => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("x_vector_only read error: {e}")))?;
                x_vector_only = matches!(value.as_str(), "1" | "true" | "TRUE" | "yes");
            }
            _ => {}
        }
    }

    let ref_audio = ref_audio.ok_or_else(|| ApiError::bad_request("ref_audio is required"))?;
    let prompt_ref_text = if x_vector_only { None } else { ref_text.clone() };
    let auto_ref_text = state.auto_ref_text;
    let transcribe_path = state.transcribe_path.clone();
    let whisper_model = state.whisper_model.clone();
    let whisper_language = state.whisper_language.clone();
    let include_ref_audio = state.include_ref_audio;
    let ref_audio_b64 = if include_ref_audio {
        Some(BASE64_STANDARD.encode(&ref_audio))
    } else {
        None
    };

    let _permit = state.semaphore.acquire().await.unwrap();
    let base = state.base.clone();
    let model_dir = state.base_model_dir.clone();
    let (mut payload, used_ref_text) = tokio::task::spawn_blocking(move || {
        let tmp = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .map_err(|e| ApiError::internal(format!("failed to create temp ref wav: {e}")))?;
        std::fs::write(tmp.path(), &ref_audio)
            .map_err(|e| ApiError::internal(format!("failed to write ref wav: {e}")))?;
        let audio = AudioBuffer::load(tmp.path())
            .map_err(|e| ApiError::bad_request(format!("invalid ref_audio: {e}")))?;
        let base = base.blocking_lock();
        let prompt_ref_text = if auto_ref_text && !x_vector_only {
            Some(transcribe_ref_text(
                tmp.path(),
                &transcribe_path,
                &whisper_model,
                &whisper_language,
            )?)
        } else {
            prompt_ref_text
        };
        let prompt = base
            .create_voice_clone_prompt(&audio, prompt_ref_text.as_deref())
            .map_err(|e| ApiError::internal(format!("prompt creation failed: {e}")))?;
        let payload = prompt_to_payload(&prompt)?;
        Ok::<_, ApiError>((payload, prompt_ref_text))
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join error: {e}")))??;

    payload.ref_text = used_ref_text;
    payload.ref_audio_b64 = ref_audio_b64;

    Ok(Json(PromptResponse {
        prompt: payload,
        model_type: "base".to_string(),
        model_dir,
    }))
}

async fn voice_clone_direct(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut ref_audio: Option<Vec<u8>> = None;
    let mut text: Option<String> = None;
    let mut ref_text: Option<String> = None;
    let mut language: Option<String> = None;
    let mut options_req: Option<SynthesisOptionsRequest> = None;
    let mut x_vector_only = false;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request(format!("invalid multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "ref_audio" => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("ref_audio read error: {e}")))?;
                ref_audio = Some(bytes.to_vec());
            }
            "text" => {
                text = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::bad_request(format!("text read error: {e}")))?,
                );
            }
            "ref_text" => {
                ref_text = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::bad_request(format!("ref_text read error: {e}")))?,
                );
            }
            "language" => {
                language = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::bad_request(format!("language read error: {e}")))?,
                );
            }
            "options" => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("options read error: {e}")))?;
                options_req = Some(
                    serde_json::from_str(&value)
                        .map_err(|e| ApiError::bad_request(format!("invalid options JSON: {e}")))?,
                );
            }
            "x_vector_only" => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("x_vector_only read error: {e}")))?;
                x_vector_only = matches!(value.as_str(), "1" | "true" | "TRUE" | "yes");
            }
            _ => {}
        }
    }

    let ref_audio = ref_audio.ok_or_else(|| ApiError::bad_request("ref_audio is required"))?;
    let text = text.ok_or_else(|| ApiError::bad_request("text is required"))?;
    let language = parse_language(language)?;
    let options = options_from_request(options_req)?;
    let prompt_ref_text = if x_vector_only { None } else { ref_text };
    let auto_ref_text = state.auto_ref_text;
    let transcribe_path = state.transcribe_path.clone();
    let whisper_model = state.whisper_model.clone();
    let whisper_language = state.whisper_language.clone();

    let _permit = state.semaphore.acquire().await.unwrap();
    let base = state.base.clone();
    let ffmpeg_path = state.ffmpeg_path.clone();

    let mp3 = tokio::task::spawn_blocking(move || {
        let tmp = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .map_err(|e| ApiError::internal(format!("failed to create temp ref wav: {e}")))?;
        std::fs::write(tmp.path(), &ref_audio)
            .map_err(|e| ApiError::internal(format!("failed to write ref wav: {e}")))?;
        let audio = AudioBuffer::load(tmp.path())
            .map_err(|e| ApiError::bad_request(format!("invalid ref_audio: {e}")))?;
        let base = base.blocking_lock();
        let prompt_ref_text = if auto_ref_text && !x_vector_only {
            Some(transcribe_ref_text(
                tmp.path(),
                &transcribe_path,
                &whisper_model,
                &whisper_language,
            )?)
        } else {
            prompt_ref_text
        };
        let prompt = base
            .create_voice_clone_prompt(&audio, prompt_ref_text.as_deref())
            .map_err(|e| ApiError::internal(format!("prompt creation failed: {e}")))?;
        let audio = base
            .synthesize_voice_clone(&text, &prompt, language, Some(options))
            .map_err(|e| ApiError::internal(format!("voice clone failed: {e}")))?;
        audio_to_mp3(&audio, &ffmpeg_path)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join error: {e}")))??;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "audio/mpeg".parse().unwrap());
    Ok((StatusCode::OK, headers, mp3).into_response())
}

async fn voice_clone(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VoiceCloneRequest>,
) -> Result<Response, ApiError> {
    let language = parse_language(req.language)?;
    let options = options_from_request(req.options)?;

    let _permit = state.semaphore.acquire().await.unwrap();
    let base = state.base.clone();
    let ffmpeg_path = state.ffmpeg_path.clone();
    let text = req.text;
    let prompt_payload = req.prompt;
    let auto_ref_text = state.auto_ref_text;
    let transcribe_path = state.transcribe_path.clone();
    let whisper_model = state.whisper_model.clone();
    let whisper_language = state.whisper_language.clone();

    let mp3 = tokio::task::spawn_blocking(move || {
        let base = base.blocking_lock();
        let mut prompt_payload = prompt_payload;
        let ref_audio_b64 = prompt_payload.ref_audio_b64.take();
        let ref_text = prompt_payload.ref_text.clone();
        let want_ref_text = prompt_payload.ref_text_ids.is_some();

        let prompt = if let Some(ref_audio_b64) = ref_audio_b64 {
            let ref_audio = BASE64_STANDARD.decode(ref_audio_b64).map_err(|e| {
                ApiError::bad_request(format!("invalid ref_audio_b64: {e}"))
            })?;
            let tmp = tempfile::Builder::new()
                .suffix(".wav")
                .tempfile()
                .map_err(|e| ApiError::internal(format!("failed to create temp ref wav: {e}")))?;
            std::fs::write(tmp.path(), &ref_audio)
                .map_err(|e| ApiError::internal(format!("failed to write ref wav: {e}")))?;
            let audio = AudioBuffer::load(tmp.path())
                .map_err(|e| ApiError::bad_request(format!("invalid ref_audio: {e}")))?;
            let prompt_ref_text = if !want_ref_text {
                None
            } else if auto_ref_text {
                Some(transcribe_ref_text(
                    tmp.path(),
                    &transcribe_path,
                    &whisper_model,
                    &whisper_language,
                )?)
            } else {
                ref_text
            };
            if want_ref_text && prompt_ref_text.is_none() {
                payload_to_prompt(prompt_payload, base.device())?
            } else {
                base.create_voice_clone_prompt(&audio, prompt_ref_text.as_deref())
                    .map_err(|e| ApiError::internal(format!("prompt creation failed: {e}")))?
            }
        } else {
            payload_to_prompt(prompt_payload, base.device())?
        };
        let audio = base
            .synthesize_voice_clone(&text, &prompt, language, Some(options))
            .map_err(|e| ApiError::internal(format!("voice clone failed: {e}")))?;
        audio_to_mp3(&audio, &ffmpeg_path)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join error: {e}")))??;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "audio/mpeg".parse().unwrap());
    Ok((StatusCode::OK, headers, mp3).into_response())
}

async fn voice_design(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VoiceDesignRequest>,
) -> Result<Response, ApiError> {
    let language = parse_language(req.language)?;
    let mut options = options_from_request(req.options)?;
    apply_min_duration(&mut options, state.voice_design_min_duration);

    let _permit = state.semaphore.acquire().await.unwrap();
    let model = state.voice_design.clone();
    let ffmpeg_path = state.ffmpeg_path.clone();
    let text = req.text;
    let instruct = req.instruct;

    let mp3 = tokio::task::spawn_blocking(move || {
        let model = model.blocking_lock();
        let audio = model
            .synthesize_voice_design(&text, &instruct, language, Some(options))
            .map_err(|e| ApiError::internal(format!("voice design failed: {e}")))?;
        audio_to_mp3(&audio, &ffmpeg_path)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join error: {e}")))??;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "audio/mpeg".parse().unwrap());
    Ok((StatusCode::OK, headers, mp3).into_response())
}

fn load_model(
    model_dir: &str,
    tokenizer_dir: Option<&str>,
    device: &candle_core::Device,
) -> Result<Qwen3TTS, ApiError> {
    Qwen3TTS::from_pretrained_with_tokenizer(model_dir, tokenizer_dir, device.clone())
        .map_err(|e| ApiError::internal(format!("failed to load model {model_dir}: {e}")))
}

#[tokio::main]
async fn main() -> Result<(), ApiError> {
    let base_model_dir = std::env::var("BASE_MODEL_DIR")
        .map_err(|_| ApiError::bad_request("BASE_MODEL_DIR is required"))?;
    let voice_design_model_dir = std::env::var("VOICE_DESIGN_MODEL_DIR")
        .map_err(|_| ApiError::bad_request("VOICE_DESIGN_MODEL_DIR is required"))?;
    let device_str = std::env::var("DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let tokenizer_dir = std::env::var("TOKENIZER_DIR").ok();
    let ffmpeg_path = std::env::var("FFMPEG_PATH").unwrap_or_else(|_| "ffmpeg".to_string());
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()
        .map_err(|e| ApiError::bad_request(format!("invalid PORT: {e}")))?;
    let max_concurrency: usize = std::env::var("MAX_CONCURRENCY")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .map_err(|e| ApiError::bad_request(format!("invalid MAX_CONCURRENCY: {e}")))?;
    let auto_ref_text = parse_env_bool("AUTO_REF_TEXT");
    let transcribe_path = std::env::var("TRANSCRIBE_PATH")
        .unwrap_or_else(|_| "/usr/local/bin/transcribe.py".to_string());
    let whisper_model = std::env::var("WHISPER_MODEL").unwrap_or_else(|_| "tiny".to_string());
    let whisper_language =
        std::env::var("WHISPER_LANGUAGE").unwrap_or_else(|_| "en".to_string());
    let include_ref_audio = parse_env_bool("PROMPT_INCLUDE_REF_AUDIO");
    let voice_design_min_duration = std::env::var("VOICE_DESIGN_MIN_DURATION_SECONDS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0);

    let device = parse_device(&device_str)
        .map_err(|e| ApiError::internal(format!("failed to init device: {e}")))?;
    tracing_subscriber::fmt::init();
    tracing::info!("Loading models on {}", device_info(&device));

    let base = Arc::new(Mutex::new(load_model(
        &base_model_dir,
        tokenizer_dir.as_deref(),
        &device,
    )?));
    let voice_design_model = Arc::new(Mutex::new(load_model(
        &voice_design_model_dir,
        tokenizer_dir.as_deref(),
        &device,
    )?));

    {
        let base_guard = base.lock().await;
        if base_guard.model_type() != Some(&ModelType::Base) {
            return Err(ApiError::bad_request("BASE_MODEL_DIR must point to a Base model"));
        }
    }
    {
        let voice_guard = voice_design_model.lock().await;
        if voice_guard.model_type() != Some(&ModelType::VoiceDesign) {
            return Err(ApiError::bad_request(
                "VOICE_DESIGN_MODEL_DIR must point to a VoiceDesign model",
            ));
        }
    }

    let state = Arc::new(AppState {
        base,
        voice_design: voice_design_model,
        base_model_dir,
        _voice_design_model_dir: voice_design_model_dir,
        ffmpeg_path,
        auto_ref_text,
        transcribe_path,
        whisper_model,
        whisper_language,
        include_ref_audio,
        voice_design_min_duration,
        semaphore: Arc::new(Semaphore::new(max_concurrency)),
    });

    let app = Router::new()
        .route("/v1/audio/voice-clone/prompt", post(create_prompt))
        .route("/v1/audio/voice-clone/prompted", post(voice_clone))
        .route("/v1/audio/voice-clone", post(voice_clone_direct))
        .route("/v1/audio/voice-design", post(voice_design))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| ApiError::bad_request(format!("invalid HOST/PORT: {e}")))?;
    tracing::info!("Listening on http://{}", addr);

    axum::serve(
        tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| ApiError::internal(format!("failed to bind: {e}")))?,
        app,
    )
    .await
    .map_err(|e| ApiError::internal(format!("server error: {e}")))?;

    Ok(())
}
