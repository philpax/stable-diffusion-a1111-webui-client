use std::{
    collections::HashMap,
    future::{Future, IntoFuture},
    pin::Pin,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

pub use image::DynamicImage;

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("invalid url; make sure it starts with http")]
    InvalidUrl,
    #[error("Not authenticated")]
    NotAuthenticated,
    #[error("invalid response body (expected {expected:?})")]
    InvalidResponse { expected: String },

    #[error("reqwest error")]
    ReqwestError(#[from] reqwest::Error),
    #[error("serde json error")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("base64 decode error")]
    Base64DecodeError(#[from] base64::DecodeError),
    #[error("image error")]
    ImageError(#[from] image::ImageError),
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
pub type Result<T> = core::result::Result<T, ClientError>;

pub struct Client {
    client: RequestClient,
    config: Config,
}
impl Client {
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
                                label: props.get("label")?.as_str()?.to_owned(),
                            },
                        )),
                        "radio" => Some((
                            id,
                            ConfigComponent::Radio {
                                choices: extract_string_array(props.get("choices")?)?,
                                id: props.get("elem_id")?.as_str()?.to_owned(),
                                label: props.get("label")?.as_str()?.to_owned(),
                            },
                        )),
                        _ => None,
                    }
                })
                .collect(),
        );

        Ok(Self { client, config })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn generate_image_from_text(&self, prompt: &str) -> GenerationTask {
        let prompt = prompt.to_owned();
        let client = self.client.clone();
        GenerationTask {
            handle: tokio::task::spawn(async move {
                #[derive(Serialize)]
                struct Request {
                    prompt: String,
                }

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
                    subseed_strength: u32,
                    width: u32,
                    height: u32,
                    sampler: String,
                    steps: usize,
                }

                let response: Response =
                    client.post("sdapi/v1/txt2img", &Request { prompt }).await?;

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
                        sampler: raw.sampler,
                        steps: raw.steps,
                    }
                };

                Ok(GenerationResult { images, info })
            }),
            client: self.client.clone(),
        }
    }
}

pub struct GenerationTask {
    handle: tokio::task::JoinHandle<Result<GenerationResult>>,
    client: RequestClient,
}
impl IntoFuture for GenerationTask {
    type Output = Result<GenerationResult>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<GenerationResult>>>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.block())
    }
}
impl GenerationTask {
    pub async fn block(self) -> Result<GenerationResult> {
        self.handle.await?
    }

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
            eta_seconds: response.eta_relative,
            progress_factor: response.progress,
        })
    }
}

pub struct GenerationProgress {
    /// Estimated time to completion, in seconds
    pub eta_seconds: f32,
    /// How much of the generation is complete, from 0 to 1
    pub progress_factor: f32,
}
impl GenerationProgress {
    pub fn is_finished(&self) -> bool {
        self.progress_factor >= 1.0
    }
}

pub struct GenerationResult {
    pub images: Vec<DynamicImage>,
    pub info: GenerationInfo,
}

#[derive(Debug)]
pub struct GenerationInfo {
    pub prompts: Vec<String>,
    pub negative_prompt: String,
    pub seeds: Vec<u64>,
    pub subseeds: Vec<u64>,
    pub subseed_strength: u32,
    pub width: u32,
    pub height: u32,
    pub sampler: String,
    pub steps: usize,
}

#[derive(Debug)]
pub enum ConfigComponent {
    Dropdown {
        choices: Vec<String>,
        id: String,
        label: String,
    },
    Radio {
        choices: Vec<String>,
        id: String,
        label: String,
    },
}

#[derive(Debug)]
pub struct Config(HashMap<u32, ConfigComponent>);
impl Config {
    pub fn checkpoints(&self) -> Result<Vec<String>> {
        self.get_dropdown("setting_sd_model_checkpoint")
    }
    pub fn embeddings(&self) -> Result<Vec<String>> {
        self.get_dropdown("train_embedding")
    }
    pub fn hypernetwork(&self) -> Result<Vec<String>> {
        self.get_dropdown("setting_sd_hypernetwork")
    }
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
