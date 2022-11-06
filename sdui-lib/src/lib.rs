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
    /// There was an error with the request.
    #[error("invalid response body (expected {expected:?})")]
    InvalidResponse {
        /// The message associated with the error.
        expected: String,
    },

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

/// Interface to the web UI.
pub struct Client {
    client: RequestClient,
    config: Config,
}
impl Client {
    /// Creates a new `Client` and authenticates to the web UI.
    pub async fn new(url: &str, authentication: Option<(&str, &str)>) -> Result<Self> {
        let client = RequestClient::new(url).await?;

        let mut body = HashMap::new();
        if let Some((username, password)) = authentication {
            body.insert("username", username);
            body.insert("password", password);
        }
        client.post_raw("login").form(&body).send().await?;

        let config = Config(
            client
                .get::<HashMap<String, serde_json::Value>>("config")
                .await?
                .get("components")
                .ok_or_else(|| ClientError::invalid_response("components"))?
                .as_array()
                .ok_or_else(|| ClientError::invalid_response("components to be an array"))?
                .into_iter()
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
        );

        Ok(Self { client, config })
    }

    /// The configuration for the web UI; retrieved during [Self::new].
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Generates an image from the provided `request`.
    ///
    /// The `GenerationTask` can be `await`ed, or its [GenerationTask::progress]
    /// can be retrieved to find out what the status of the generation is.
    pub fn generate_image_from_text(&self, request: &GenerationRequest) -> GenerationTask {
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

        let (seed_resize_from_w, seed_resize_from_h) =
            (request.seed_resize_from_w, request.seed_resize_from_h);

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
                all_prompts: Vec<String>,
                negative_prompt: String,
                all_seeds: Vec<u64>,
                all_subseeds: Vec<u64>,
                subseed_strength: f32,
                width: u32,
                height: u32,
                sampler: String,
                steps: usize,
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
                    negative_prompt: raw.negative_prompt,
                    seeds: raw.all_seeds,
                    subseeds: raw.all_subseeds,
                    subseed_strength: raw.subseed_strength,
                    width: raw.width,
                    height: raw.height,
                    sampler: Sampler::try_from(raw.sampler.as_str()).unwrap(),
                    steps: raw.steps,

                    firstphase_width: request.firstphase_width,
                    firstphase_height: request.firstphase_height,
                    cfg_scale: request.cfg_scale,
                    denoising_strength: request.denoising_strength,
                    eta: request.eta,
                    tiling: request.tiling,
                    enable_hr: request.enable_hr,
                    restore_faces: request.restore_faces,
                    s_churn: request.s_churn,
                    s_noise: request.s_noise,
                    s_tmax: request.s_tmax,
                    s_tmin: request.s_tmin,
                    seed_resize_from_w,
                    seed_resize_from_h,
                    styles: request.styles,
                }
            };

            Ok(GenerationResult { images, info })
        });

        GenerationTask {
            handle,
            client: self.client.clone(),
        }
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
    /// The negative prompt that was applied to each image.
    pub negative_prompt: String,
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

#[derive(Debug)]
enum ConfigComponent {
    Dropdown { choices: Vec<String>, id: String },
    Radio { choices: Vec<String>, id: String },
}

/// The configuration for the Web UI.
#[derive(Debug)]
pub struct Config(HashMap<u32, ConfigComponent>);
impl Config {
    /// All of the available checkpoints/models available.
    pub fn checkpoints(&self) -> Result<Vec<String>> {
        self.get_dropdown("setting_sd_model_checkpoint")
    }
    /// All of the embeddings available.
    pub fn embeddings(&self) -> Result<Vec<String>> {
        self.get_dropdown("train_embedding")
    }
    /// All of the hypernetworks available.
    pub fn hypernetworks(&self) -> Result<Vec<String>> {
        self.get_dropdown("setting_sd_hypernetwork")
    }
    /// All of the samplers available for text to image generation.
    pub fn txt2img_samplers(&self) -> Result<Vec<String>> {
        self.get_radio("txt2img_sampling")
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
}
impl RequestClient {
    async fn new(url: &str) -> Result<Self> {
        if !url.starts_with("http") {
            return Err(ClientError::InvalidUrl);
        }

        let url = url.strip_suffix("/").unwrap_or(url).to_owned();
        let client = reqwest::ClientBuilder::new().cookie_store(true).build()?;

        Ok(Self { url, client })
    }

    fn url(&self, endpoint: &str) -> String {
        format!("{}/{}", self.url, endpoint)
    }

    fn check_for_authentication<R: DeserializeOwned>(body: String) -> Result<R> {
        let json_body: HashMap<String, serde_json::Value> = serde_json::from_str(&body)?;
        match json_body.get("detail") {
            Some(serde_json::Value::String(payload)) if payload == "Not authenticated" => {
                Err(ClientError::NotAuthenticated)
            }
            _ => Ok(serde_json::from_str(&body)?),
        }
    }

    async fn send<R: DeserializeOwned>(builder: reqwest::RequestBuilder) -> Result<R> {
        Self::check_for_authentication(builder.send().await?.text().await?)
    }
    async fn get<R: DeserializeOwned>(&self, endpoint: &str) -> Result<R> {
        Self::send(self.client.get(self.url(endpoint))).await
    }
    async fn post<R: DeserializeOwned, T: Serialize>(&self, endpoint: &str, body: &T) -> Result<R> {
        Self::send(self.client.post(self.url(endpoint)).json(body)).await
    }
    fn post_raw(&self, endpoint: &str) -> reqwest::RequestBuilder {
        self.client.post(self.url(endpoint))
    }
}

fn extract_string_array(value: &serde_json::Value) -> Option<Vec<String>> {
    Some(
        value
            .as_array()?
            .into_iter()
            .flat_map(|s| Some(s.as_str()?.to_owned()))
            .collect(),
    )
}
