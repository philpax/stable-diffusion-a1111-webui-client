#![deny(missing_docs)]
//! This is a client for the Automatic1111 stable-diffusion web UI.

use std::{
    collections::HashMap,
    future::{Future, IntoFuture},
    pin::Pin,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

pub use image::DynamicImage;

/// All potential errors that the client can produce.
#[derive(Error, Debug)]
pub enum ClientError {
    /// The URL passed to the `Client` was invalid.
    #[error("invalid url; make sure it starts with http")]
    InvalidUrl,
    /// The credentials for the `Client` are wrong or missing.
    #[error("Not authenticated")]
    NotAuthenticated,
    /// There was an arbitrary error while servicing the request.
    #[error("error: {message}")]
    Error {
        /// The message associated with the error.
        message: String,
    },
    /// The response body was missing some data.
    #[error("invalid response body (expected {expected:?})")]
    InvalidResponse {
        /// The data that was expected to be there, but wasn't.
        expected: String,
    },
    /// The UI experienced an internal server error.
    #[error("internal server error")]
    InternalServerError,
    /// The operation requires access to the SDUI's config, which is not
    /// accessible through UI auth alone
    #[error("Config not available")]
    ConfigNotAvailable,

    /// Error returned by `reqwest`.
    #[error("reqwest error")]
    ReqwestError(#[from] reqwest::Error),
    /// Error returned by `serde_json`.
    #[error("serde json error")]
    SerdeJsonError(#[from] serde_json::Error),
    /// Error returned by `base64`.
    #[error("base64 decode error")]
    Base64DecodeError(#[from] base64::DecodeError),
    /// Error returned by `image`.
    #[error("image error")]
    ImageError(#[from] image::ImageError),
    /// Error returned by `tokio` due to a failure to join the task.
    #[error("tokio join error")]
    TokioJoinError(#[from] tokio::task::JoinError),
    /// Error returned by `chrono` when trying to parse a datetime.
    #[error("chrono parse error")]
    ChronoParseError(#[from] chrono::format::ParseError),
}
impl ClientError {
    fn invalid_response(expected: &str) -> Self {
        Self::InvalidResponse {
            expected: expected.to_string(),
        }
    }
}
/// Result type for the `Client`.
pub type Result<T> = core::result::Result<T, ClientError>;

/// The type of authentication to use with a [Client].
pub enum Authentication<'a> {
    /// The server is unauthenticated
    None,
    /// The server is using API authentication (Authorization header)
    ApiAuth(&'a str, &'a str),
    /// The server is using Gradio authentication (/login)
    GradioAuth(&'a str, &'a str),
}

/// Interface to the web UI.
pub struct Client {
    client: RequestClient,
    config: Option<Config>,
}
impl Client {
    /// Creates a new `Client` and authenticates to the web UI.
    pub async fn new(url: &str, authentication: Authentication<'_>) -> Result<Self> {
        let mut client = RequestClient::new(url).await?;

        let mut ui_auth = true;
        match authentication {
            Authentication::None => {}
            Authentication::ApiAuth(username, password) => {
                client.set_authentication_token(base64::encode(format!("{username}:{password}")));
                ui_auth = false;
            }
            Authentication::GradioAuth(username, password) => {
                client
                    .post_raw("login")
                    .form(&HashMap::<&str, &str>::from_iter([
                        ("username", username),
                        ("password", password),
                    ]))
                    .send()
                    .await?;
            }
        }

        // The config is only available if there is no authentication or the authentication
        // occurred with Gradio; otherwise, API auth is insufficient to retrieve the config
        let config = if ui_auth {
            Some(Config::new(&client).await?)
        } else {
            None
        };

        Ok(Self { client, config })
    }

    /// Generates an image from the provided `request`.
    ///
    /// The `GenerationTask` can be `await`ed, or its [GenerationTask::progress]
    /// can be retrieved to find out what the status of the generation is.
    pub fn generate_image_from_text(&self, request: &GenerationRequest) -> Result<GenerationTask> {
        #[derive(Serialize)]
        struct OptionsRequest {
            sd_model_checkpoint: String,
        }

        #[derive(Serialize)]
        struct Request {
            batch_size: i32,
            cfg_scale: f32,
            denoising_strength: f32,
            enable_hr: bool,
            eta: f32,
            firstphase_height: u32,
            firstphase_width: u32,
            height: u32,
            n_iter: u32,
            negative_prompt: String,
            prompt: String,
            restore_faces: bool,
            s_churn: f32,
            s_noise: f32,
            s_tmax: f32,
            s_tmin: f32,
            sampler_index: String,
            seed: i64,
            seed_resize_from_h: i32,
            seed_resize_from_w: i32,
            steps: u32,
            styles: Vec<String>,
            subseed: i64,
            subseed_strength: f32,
            tiling: bool,
            width: u32,
        }

        let options_request = request.model.map(|s| OptionsRequest {
            sd_model_checkpoint: s.title.clone(),
        });

        let request = {
            let d = Request {
                enable_hr: false,
                denoising_strength: 0.0,
                firstphase_width: 0,
                firstphase_height: 0,
                prompt: String::new(),
                styles: vec![],
                seed: -1,
                subseed: -1,
                subseed_strength: 0.0,
                seed_resize_from_h: -1,
                seed_resize_from_w: -1,
                batch_size: 1,
                n_iter: 1,
                steps: 50,
                cfg_scale: 7.0,
                width: 512,
                height: 512,
                restore_faces: false,
                tiling: false,
                negative_prompt: String::new(),
                eta: 0.0,
                s_churn: 0.0,
                s_tmax: 0.0,
                s_tmin: 0.0,
                s_noise: 1.0,
                sampler_index: "Euler".to_owned(),
            };
            let r = request;
            Request {
                batch_size: r.batch_size.map(|i| i as i32).unwrap_or(d.batch_size),
                cfg_scale: r.cfg_scale.unwrap_or(d.cfg_scale),
                denoising_strength: r.denoising_strength.unwrap_or(d.denoising_strength),
                enable_hr: r.enable_hr.unwrap_or(d.enable_hr),
                eta: r.eta.unwrap_or(d.eta),
                firstphase_height: r.firstphase_height.unwrap_or(d.firstphase_height),
                firstphase_width: r.firstphase_width.unwrap_or(d.firstphase_width),
                height: r.height.unwrap_or(d.height),
                n_iter: r.batch_count.unwrap_or(d.n_iter),
                negative_prompt: r
                    .negative_prompt
                    .map(|s| s.to_owned())
                    .unwrap_or(d.negative_prompt),
                prompt: r.prompt.to_owned(),
                restore_faces: r.restore_faces.unwrap_or(d.restore_faces),
                s_churn: r.s_churn.unwrap_or(d.s_churn),
                s_noise: r.s_noise.unwrap_or(d.s_noise),
                s_tmax: r.s_tmax.unwrap_or(d.s_tmax),
                s_tmin: r.s_tmin.unwrap_or(d.s_tmin),
                sampler_index: r.sampler.map(|s| s.to_string()).unwrap_or(d.sampler_index),
                seed: r.seed.unwrap_or(d.seed),
                seed_resize_from_h: r
                    .seed_resize_from_h
                    .map(|i| i as i32)
                    .unwrap_or(d.seed_resize_from_h),
                seed_resize_from_w: r
                    .seed_resize_from_w
                    .map(|i| i as i32)
                    .unwrap_or(d.seed_resize_from_w),
                steps: r.steps.unwrap_or(d.steps),
                styles: r.styles.clone().unwrap_or(d.styles),
                subseed: r.subseed.unwrap_or(d.subseed),
                subseed_strength: r.subseed_strength.unwrap_or(d.subseed_strength),
                tiling: r.tiling.unwrap_or(d.tiling),
                width: r.width.unwrap_or(d.width),
            }
        };

        let client = self.client.clone();
        let handle = tokio::task::spawn(async move {
            #[derive(Deserialize)]
            struct Response {
                images: Vec<String>,
                info: String,
            }

            #[derive(Deserialize)]
            pub struct InfoResponse {
                all_negative_prompts: Vec<String>,
                all_prompts: Vec<String>,

                all_seeds: Vec<u64>,
                seed_resize_from_h: i32,
                seed_resize_from_w: i32,

                all_subseeds: Vec<u64>,
                subseed_strength: f32,

                cfg_scale: f32,
                clip_skip: usize,
                denoising_strength: f32,
                face_restoration_model: Option<String>,
                is_using_inpainting_conditioning: bool,
                job_timestamp: String,
                restore_faces: bool,
                sd_model_hash: String,
                styles: Vec<String>,

                width: u32,
                height: u32,
                sampler_name: String,
                steps: usize,
            }

            if let Some(options_request) = options_request {
                // Used to set the model if requested
                client.post("sdapi/v1/options", &options_request).await?;
            }

            let response: Response = client.post("sdapi/v1/txt2img", &request).await?;
            let images = response
                .images
                .iter()
                .map(|b64| Ok(image::load_from_memory(&base64::decode(b64)?)?))
                .collect::<Result<Vec<_>>>()?;
            let info = {
                let raw: InfoResponse = serde_json::from_str(&response.info)?;
                GenerationInfo {
                    prompts: raw.all_prompts,
                    negative_prompts: raw.all_negative_prompts,
                    seeds: raw.all_seeds,
                    subseeds: raw.all_subseeds,
                    subseed_strength: raw.subseed_strength,
                    width: raw.width,
                    height: raw.height,
                    sampler: Sampler::try_from(raw.sampler_name.as_str()).unwrap(),
                    steps: raw.steps,

                    firstphase_width: request.firstphase_width,
                    firstphase_height: request.firstphase_height,
                    cfg_scale: raw.cfg_scale,
                    denoising_strength: raw.denoising_strength,
                    eta: request.eta,
                    tiling: request.tiling,
                    enable_hr: request.enable_hr,
                    restore_faces: raw.restore_faces,
                    s_churn: request.s_churn,
                    s_noise: request.s_noise,
                    s_tmax: request.s_tmax,
                    s_tmin: request.s_tmin,
                    seed_resize_from_w: Some(raw.seed_resize_from_w)
                        .filter(|v| *v > 0)
                        .map(|v| v as u32),
                    seed_resize_from_h: Some(raw.seed_resize_from_h)
                        .filter(|v| *v > 0)
                        .map(|v| v as u32),
                    styles: raw.styles,

                    clip_skip: raw.clip_skip,
                    face_restoration_model: raw.face_restoration_model,
                    is_using_inpainting_conditioning: raw.is_using_inpainting_conditioning,
                    job_timestamp: chrono::NaiveDateTime::parse_from_str(
                        &raw.job_timestamp,
                        "%Y%m%d%H%M%S",
                    )?
                    .and_local_timezone(chrono::Local)
                    .unwrap(),
                    model_hash: raw.sd_model_hash,
                }
            };

            Ok(GenerationResult { images, info })
        });

        Ok(GenerationTask {
            handle,
            client: self.client.clone(),
        })
    }

    /// Get the embeddings
    pub async fn embeddings(&self) -> Result<Vec<String>> {
        if let Some(config) = &self.config {
            config.embeddings()
        } else {
            Err(ClientError::ConfigNotAvailable)
        }
    }

    /// Get the options
    pub async fn options(&self) -> Result<Options> {
        #[derive(Serialize, Deserialize)]
        struct OptionsRaw {
            s_churn: f32,
            s_noise: f32,
            s_tmin: f32,
            sd_hypernetwork: String,
            sd_model_checkpoint: String,
        }

        self.client
            .get::<OptionsRaw>("sdapi/v1/options")
            .await
            .map(|r| Options {
                hypernetwork: r.sd_hypernetwork,
                model: r.sd_model_checkpoint,
                s_churn: r.s_churn,
                s_noise: r.s_noise,
                s_tmin: r.s_tmin,
            })
    }

    /// Get the samplers
    pub async fn samplers(&self) -> Result<Vec<Sampler>> {
        #[derive(Serialize, Deserialize)]
        struct SamplerRaw {
            aliases: Vec<String>,
            name: String,
        }

        self.client
            .get::<Vec<SamplerRaw>>("sdapi/v1/samplers")
            .await
            .map(|r| {
                r.into_iter()
                    .filter_map(|s| Sampler::try_from(s.name.as_str()).ok())
                    .collect::<Vec<_>>()
            })
    }

    /// Get the upscalers
    pub async fn upscalers(&self) -> Result<Vec<Upscaler>> {
        #[derive(Serialize, Deserialize)]
        struct UpscalerRaw {
            model_name: Option<String>,
            model_path: Option<String>,
            model_url: Option<String>,
            name: String,
        }

        self.client
            .get::<Vec<UpscalerRaw>>("sdapi/v1/upscalers")
            .await
            .map(|r| {
                r.into_iter()
                    .map(|r| Upscaler {
                        name: r.name,
                        model_name: r.model_name,
                    })
                    .collect()
            })
    }

    /// Get the models
    pub async fn models(&self) -> Result<Vec<Model>> {
        #[derive(Serialize, Deserialize)]
        struct ModelRaw {
            config: String,
            filename: String,
            hash: String,
            model_name: String,
            title: String,
        }

        self.client
            .get::<Vec<ModelRaw>>("sdapi/v1/sd-models")
            .await
            .map(|r| {
                r.into_iter()
                    .map(|r| Model {
                        title: r.title,
                        name: r.model_name,
                    })
                    .collect()
            })
    }

    /// Get the hypernetworks
    pub async fn hypernetworks(&self) -> Result<Vec<String>> {
        #[derive(Serialize, Deserialize)]
        struct HypernetworkRaw {
            name: String,
            path: String,
        }

        self.client
            .get::<Vec<HypernetworkRaw>>("sdapi/v1/hypernetworks")
            .await
            .map(|r| r.into_iter().map(|s| s.name).collect())
    }

    /// Get the face restorers
    pub async fn face_restorers(&self) -> Result<Vec<String>> {
        #[derive(Serialize, Deserialize)]
        struct FaceRestorerRaw {
            cmd_dir: Option<String>,
            name: String,
        }

        self.client
            .get::<Vec<FaceRestorerRaw>>("sdapi/v1/face-restorers")
            .await
            .map(|r| r.into_iter().map(|s| s.name).collect())
    }

    /// Get the real ESRGAN models
    pub async fn realesrgan_models(&self) -> Result<Vec<String>> {
        #[derive(Serialize, Deserialize)]
        struct RealEsrganModelRaw {
            name: String,
            path: Option<String>,
            scale: i64,
        }

        self.client
            .get::<Vec<RealEsrganModelRaw>>("sdapi/v1/realesrgan-models")
            .await
            .map(|r| r.into_iter().map(|s| s.name).collect())
    }

    /// Get the prompt styles
    pub async fn prompt_styles(&self) -> Result<Vec<PromptStyle>> {
        #[derive(Serialize, Deserialize)]
        struct PromptStyleRaw {
            name: String,
            negative_prompt: Option<String>,
            prompt: Option<String>,
        }

        self.client
            .get::<Vec<PromptStyleRaw>>("sdapi/v1/prompt-styles")
            .await
            .map(|r| {
                r.into_iter()
                    .map(|r| PromptStyle {
                        name: r.name,
                        prompt: r.prompt,
                        negative_prompt: r.negative_prompt,
                    })
                    .collect()
            })
    }

    /// Get the artist categories
    pub async fn artist_categories(&self) -> Result<Vec<String>> {
        self.client
            .get::<Vec<String>>("sdapi/v1/artist-categories")
            .await
    }

    /// Get the artists
    pub async fn artists(&self) -> Result<Vec<Artist>> {
        #[derive(Serialize, Deserialize)]
        struct ArtistRaw {
            category: String,
            name: String,
            score: f32,
        }

        self.client
            .get::<Vec<ArtistRaw>>("sdapi/v1/artists")
            .await
            .map(|r| {
                r.into_iter()
                    .map(|r| Artist {
                        name: r.name,
                        category: r.category,
                    })
                    .collect()
            })
    }
}

/// Represents an ongoing generation.
pub struct GenerationTask {
    handle: tokio::task::JoinHandle<Result<GenerationResult>>,
    client: RequestClient,
}
impl IntoFuture for GenerationTask {
    type Output = Result<GenerationResult>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<GenerationResult>> + Send + Sync>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.block())
    }
}
impl GenerationTask {
    /// Waits for the generation to be complete.
    pub async fn block(self) -> Result<GenerationResult> {
        self.handle.await?
    }

    /// Retrieves the progress of the ongoing generation.
    pub async fn progress(&self) -> Result<GenerationProgress> {
        if self.handle.is_finished() {
            return Ok(GenerationProgress {
                eta_seconds: 0.0,
                progress_factor: 1.0,
            });
        }

        #[derive(Deserialize)]
        struct Response {
            eta_relative: f32,
            progress: f32,
        }

        let response: Response = self.client.get("sdapi/v1/progress").await?;
        Ok(GenerationProgress {
            eta_seconds: response.eta_relative.max(0.0),
            progress_factor: response.progress.clamp(0.0, 1.0),
        })
    }
}

/// How much of the generation is complete.
///
/// Note that this can return a zero value on completion as the web UI
/// can take time to return the result after completion.
pub struct GenerationProgress {
    /// Estimated time to completion, in seconds
    pub eta_seconds: f32,
    /// How much of the generation is complete, from 0 to 1
    pub progress_factor: f32,
}
impl GenerationProgress {
    /// Whether or not the generation has completed.
    pub fn is_finished(&self) -> bool {
        self.progress_factor >= 1.0
    }
}

/// The parameters to the generation.
///
/// Consider using the [Default] trait to fill in the
/// parameters that you don't need to fill in.
#[derive(Default)]
pub struct GenerationRequest<'a> {
    /// The prompt
    pub prompt: &'a str,
    /// The negative prompt (elements to avoid from the generation)
    pub negative_prompt: Option<&'a str>,

    /// The number of images in each batch
    pub batch_size: Option<u32>,
    /// The number of batches
    pub batch_count: Option<u32>,

    /// The width of the generated image
    pub width: Option<u32>,
    /// The height of the generated image
    pub height: Option<u32>,
    /// The width of the first phase of the generated image
    pub firstphase_width: Option<u32>,
    /// The height of the first phase of the generated image
    pub firstphase_height: Option<u32>,

    /// The Classifier-Free Guidance scale; how strongly the prompt is
    /// applied to the generation
    pub cfg_scale: Option<f32>,
    /// The denoising strength
    pub denoising_strength: Option<f32>,
    /// The η parameter
    pub eta: Option<f32>,
    /// The sampler to use for the generation
    pub sampler: Option<Sampler>,
    /// The number of steps
    pub steps: Option<u32>,
    /// The model override to use. If not supplied, the currently-set model will be used.
    pub model: Option<&'a Model>,

    /// Whether or not the image should be tiled at the edges
    pub tiling: Option<bool>,
    /// Unknown
    pub enable_hr: Option<bool>,
    /// Whether or not to apply the face restoration
    pub restore_faces: Option<bool>,

    /// s_churn
    pub s_churn: Option<f32>,
    /// s_noise
    pub s_noise: Option<f32>,
    /// s_tmax
    pub s_tmax: Option<f32>,
    /// s_tmin
    pub s_tmin: Option<f32>,

    /// The seed to use for this generation. This will apply to the first image,
    /// and the web UI will generate the successive seeds.
    pub seed: Option<i64>,
    /// The width to resize the image from if reusing a seed with a different size
    pub seed_resize_from_w: Option<u32>,
    /// The height to resize the image from if reusing a seed with a different size
    pub seed_resize_from_h: Option<u32>,
    /// The subseed to use for this generation
    pub subseed: Option<i64>,
    /// The strength of the subseed
    pub subseed_strength: Option<f32>,

    /// Any styles to apply to the generation
    pub styles: Option<Vec<String>>,
}

/// The result of the generation.
pub struct GenerationResult {
    /// The images produced by the generator.
    pub images: Vec<DynamicImage>,
    /// The information associated with this generation.
    pub info: GenerationInfo,
}

/// The information associated with a generation.
#[derive(Debug)]
pub struct GenerationInfo {
    /// The prompts used for each image in the generation.
    pub prompts: Vec<String>,
    /// The negative prompt for each image in the generation.
    pub negative_prompts: Vec<String>,
    /// The seeds for the images; each seed corresponds to an image.
    pub seeds: Vec<u64>,
    /// The subseeds for the images; each seed corresponds to an image.
    pub subseeds: Vec<u64>,
    /// The strength of the subseed.
    pub subseed_strength: f32,
    /// The width of the generated images.
    pub width: u32,
    /// The height of the generated images.
    pub height: u32,
    /// The sampler that was used for this generation.
    pub sampler: Sampler,
    /// The number of steps that were used for each generation.
    pub steps: usize,

    /// The width of the first phase of the generated image
    pub firstphase_width: u32,
    /// The height of the first phase of the generated image
    pub firstphase_height: u32,

    /// The Classifier-Free Guidance scale; how strongly the prompt was
    /// applied to the generation
    pub cfg_scale: f32,
    /// The denoising strength
    pub denoising_strength: f32,
    /// The η parameter
    pub eta: f32,

    /// Whether or not the image was tiled at the edges
    pub tiling: bool,
    /// Unknown
    pub enable_hr: bool,
    /// Whether or not the face restoration was applied
    pub restore_faces: bool,

    /// s_churn
    pub s_churn: f32,
    /// s_noise
    pub s_noise: f32,
    /// s_tmax
    pub s_tmax: f32,
    /// s_tmin
    pub s_tmin: f32,

    /// The width to resize the image from if reusing a seed with a different size
    pub seed_resize_from_w: Option<u32>,
    /// The height to resize the image from if reusing a seed with a different size
    pub seed_resize_from_h: Option<u32>,

    /// Any styles applied to the generation
    pub styles: Vec<String>,

    /// CLIP rounds to skip
    pub clip_skip: usize,
    /// Face restoration model in use
    pub face_restoration_model: Option<String>,
    /// Whether or not inpainting conditioning is being used
    pub is_using_inpainting_conditioning: bool,
    /// When the job was run
    pub job_timestamp: chrono::DateTime<chrono::Local>,
    /// The hash of the model in use
    pub model_hash: String,
}

/// The sampler to use for the generation.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    /// Euler a
    EulerA,
    /// Euler
    Euler,
    /// LMS
    Lms,
    /// Heun
    Heun,
    /// DPM2
    Dpm2,
    /// DPM2 a
    Dpm2A,
    /// DPM fast
    DpmFast,
    /// DPM adaptive
    DpmAdaptive,
    /// LMS Karras
    LmsKarras,
    /// DPM2 Karras
    Dpm2Karras,
    /// DPM2 a Karras
    Dpm2AKarras,
    /// DDIM
    Ddim,
    /// PLMS
    Plms,
}
impl ToString for Sampler {
    fn to_string(&self) -> String {
        match self {
            Sampler::EulerA => "Euler a",
            Sampler::Euler => "Euler",
            Sampler::Lms => "LMS",
            Sampler::Heun => "Heun",
            Sampler::Dpm2 => "DPM2",
            Sampler::Dpm2A => "DPM2 a",
            Sampler::DpmFast => "DPM fast",
            Sampler::DpmAdaptive => "DPM adaptive",
            Sampler::LmsKarras => "LMS Karras",
            Sampler::Dpm2Karras => "DPM2 Karras",
            Sampler::Dpm2AKarras => "DPM2 a Karras",
            Sampler::Ddim => "DDIM",
            Sampler::Plms => "PLMS",
        }
        .to_string()
    }
}
impl TryFrom<&str> for Sampler {
    type Error = ();

    fn try_from(s: &str) -> core::result::Result<Sampler, ()> {
        match s {
            "Euler a" => Ok(Sampler::EulerA),
            "Euler" => Ok(Sampler::Euler),
            "LMS" => Ok(Sampler::Lms),
            "Heun" => Ok(Sampler::Heun),
            "DPM2" => Ok(Sampler::Dpm2),
            "DPM2 a" => Ok(Sampler::Dpm2A),
            "DPM fast" => Ok(Sampler::DpmFast),
            "DPM adaptive" => Ok(Sampler::DpmAdaptive),
            "LMS Karras" => Ok(Sampler::LmsKarras),
            "DPM2 Karras" => Ok(Sampler::Dpm2Karras),
            "DPM2 a Karras" => Ok(Sampler::Dpm2AKarras),
            "DDIM" => Ok(Sampler::Ddim),
            "PLMS" => Ok(Sampler::Plms),
            _ => Err(()),
        }
    }
}
impl Sampler {
    /// All of the possible values.
    pub const VALUES: &[Sampler] = &[
        Sampler::EulerA,
        Sampler::Euler,
        Sampler::Lms,
        Sampler::Heun,
        Sampler::Dpm2,
        Sampler::Dpm2A,
        Sampler::DpmFast,
        Sampler::DpmAdaptive,
        Sampler::LmsKarras,
        Sampler::Dpm2Karras,
        Sampler::Dpm2AKarras,
        Sampler::Ddim,
        Sampler::Plms,
    ];
}

/// The currently set options for the UI
#[derive(Debug)]
pub struct Options {
    /// Current hypernetwork
    pub hypernetwork: String,
    /// Current model
    pub model: String,

    /// s_churn
    pub s_churn: f32,
    /// s_noise
    pub s_noise: f32,
    /// s_tmin
    pub s_tmin: f32,
}

/// Upscaler
#[derive(Debug)]
pub struct Upscaler {
    /// Name of the upscaler
    pub name: String,
    /// Name of the model used for this upscaler
    pub model_name: Option<String>,
}

/// Model
#[derive(Debug)]
pub struct Model {
    /// Title of the model
    pub title: String,
    /// Name of the model
    pub name: String,
}

/// Prompt style
#[derive(Debug)]
pub struct PromptStyle {
    /// Name of the style
    pub name: String,
    /// Prompt of the style
    pub prompt: Option<String>,
    /// Negative prompt of the style
    pub negative_prompt: Option<String>,
}

/// Artist
#[derive(Debug)]
pub struct Artist {
    /// Name of the artist
    pub name: String,
    /// Category the artist belongs to
    pub category: String,
}

#[derive(Debug)]
enum ConfigComponent {
    Dropdown { choices: Vec<String>, id: String },
    Radio { choices: Vec<String>, id: String },
}

/// The configuration for the Web UI.
#[derive(Debug)]
struct Config(HashMap<u32, ConfigComponent>);
impl Config {
    async fn new(client: &RequestClient) -> Result<Self> {
        Ok(Self(
            client
                .get::<HashMap<String, serde_json::Value>>("config")
                .await?
                .get("components")
                .ok_or_else(|| ClientError::invalid_response("components"))?
                .as_array()
                .ok_or_else(|| ClientError::invalid_response("components to be an array"))?
                .iter()
                .filter_map(|v| v.as_object())
                .filter_map(|o| {
                    let id = o.get("id")?.as_u64()? as u32;
                    let comp_type = o.get("type")?.as_str()?;
                    let props = o.get("props")?.as_object()?;
                    match comp_type {
                        "dropdown" => Some((
                            id,
                            ConfigComponent::Dropdown {
                                choices: extract_string_array(props.get("choices")?)?,
                                id: props.get("elem_id")?.as_str()?.to_owned(),
                            },
                        )),
                        "radio" => Some((
                            id,
                            ConfigComponent::Radio {
                                choices: extract_string_array(props.get("choices")?)?,
                                id: props.get("elem_id")?.as_str()?.to_owned(),
                            },
                        )),
                        _ => None,
                    }
                })
                .collect(),
        ))
    }

    /// All of the embeddings available.
    fn embeddings(&self) -> Result<Vec<String>> {
        self.get_dropdown("train_embedding")
    }

    fn values(&self) -> impl Iterator<Item = &ConfigComponent> {
        self.0.values()
    }
    fn get_dropdown(&self, target_id: &str) -> Result<Vec<String>> {
        self.values()
            .find_map(|comp| match comp {
                ConfigComponent::Dropdown { id, choices, .. } if id == target_id => {
                    Some(choices.clone())
                }
                _ => None,
            })
            .ok_or_else(|| {
                ClientError::invalid_response(&format!("no {target_id} dropdown component"))
            })
    }
    #[allow(dead_code)]
    fn get_radio(&self, target_id: &str) -> Result<Vec<String>> {
        self.values()
            .find_map(|comp| match comp {
                ConfigComponent::Radio { id, choices, .. } if id == target_id => {
                    Some(choices.clone())
                }
                _ => None,
            })
            .ok_or_else(|| {
                ClientError::invalid_response(&format!("no {target_id} radio component"))
            })
    }
}

#[derive(Clone)]
struct RequestClient {
    url: String,
    client: reqwest::Client,
    authentication_token: Option<String>,
}
impl RequestClient {
    async fn new(url: &str) -> Result<Self> {
        if !url.starts_with("http") {
            return Err(ClientError::InvalidUrl);
        }

        let url = url.strip_suffix('/').unwrap_or(url).to_owned();
        let client = reqwest::ClientBuilder::new().cookie_store(true).build()?;

        Ok(Self {
            url,
            client,
            authentication_token: None,
        })
    }

    fn set_authentication_token(&mut self, token: String) {
        self.authentication_token = Some(format!("Basic {token}"));
    }

    fn url(&self, endpoint: &str) -> String {
        format!("{}/{}", self.url, endpoint)
    }

    fn check_for_authentication<R: DeserializeOwned>(body: String) -> Result<R> {
        if body.trim() == "Internal Server Error" {
            return Err(ClientError::InternalServerError);
        }

        match serde_json::from_str::<HashMap<String, serde_json::Value>>(&body) {
            Ok(json_body) => match json_body.get("detail") {
                Some(serde_json::Value::String(message)) => {
                    if message == "Not authenticated" {
                        Err(ClientError::NotAuthenticated)
                    } else {
                        Err(ClientError::Error {
                            message: message.clone(),
                        })
                    }
                }
                Some(other_error) => Err(ClientError::Error {
                    message: other_error.to_string(),
                }),
                _ => Ok(serde_json::from_str(&body)?),
            },
            Err(_) => Ok(serde_json::from_str(&body)?),
        }
    }

    async fn send<R: DeserializeOwned>(&self, builder: reqwest::RequestBuilder) -> Result<R> {
        let builder = if let Some(token) = &self.authentication_token {
            builder.header("Authorization", token)
        } else {
            builder
        };
        Self::check_for_authentication(builder.send().await?.text().await?)
    }
    async fn get<R: DeserializeOwned>(&self, endpoint: &str) -> Result<R> {
        self.send(self.client.get(self.url(endpoint))).await
    }
    async fn post<R: DeserializeOwned, T: Serialize>(&self, endpoint: &str, body: &T) -> Result<R> {
        self.send(self.client.post(self.url(endpoint)).json(body))
            .await
    }
    fn post_raw(&self, endpoint: &str) -> reqwest::RequestBuilder {
        self.client.post(self.url(endpoint))
    }
}

fn extract_string_array(value: &serde_json::Value) -> Option<Vec<String>> {
    Some(
        value
            .as_array()?
            .iter()
            .flat_map(|s| Some(s.as_str()?.to_owned()))
            .collect(),
    )
}
