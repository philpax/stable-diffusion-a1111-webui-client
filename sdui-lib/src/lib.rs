use std::collections::HashMap;

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
}
impl ClientError {
    fn invalid_response(expected: &str) -> Self {
        Self::InvalidResponse {
            expected: expected.to_string(),
        }
    }
}
pub type Result<T> = core::result::Result<T, ClientError>;

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

#[derive(Debug, Deserialize)]
pub struct GenerationInfo {
    #[serde(rename = "all_prompts")]
    pub prompts: Vec<String>,
    pub negative_prompt: String,
    #[serde(rename = "all_seeds")]
    pub seeds: Vec<u64>,
    #[serde(rename = "all_subseeds")]
    pub subseeds: Vec<u64>,
    pub subseed_strength: u32,
    pub width: u32,
    pub height: u32,
    pub sampler: String,
    pub steps: usize,
}

pub struct GenerationResult {
    pub images: Vec<DynamicImage>,
    pub info: GenerationInfo,
}

pub struct Client {
    url: String,
    client: reqwest::Client,
}
impl Client {
    pub async fn new(url: &str, authentication: Option<(&str, &str)>) -> Result<Self> {
        if !url.starts_with("http") {
            return Err(ClientError::InvalidUrl);
        }

        let url = url.strip_suffix("/").unwrap_or(url).to_owned();
        let client = reqwest::ClientBuilder::new().cookie_store(true).build()?;

        let mut body = HashMap::new();
        if let Some((username, password)) = authentication {
            body.insert("username", username);
            body.insert("password", password);
        }
        client
            .post(format!("{url}/login"))
            .form(&body)
            .send()
            .await?
            .text()
            .await?;

        Ok(Self { url, client })
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

    async fn get<R: DeserializeOwned>(&self, endpoint: &str) -> Result<R> {
        Self::check_for_authentication(
            self.client
                .get(format!("{}/{}", self.url, endpoint))
                .send()
                .await?
                .text()
                .await?,
        )
    }

    async fn post<R: DeserializeOwned, T: Serialize>(&self, endpoint: &str, body: &T) -> Result<R> {
        Self::check_for_authentication(
            self.client
                .post(format!("{}/{}", self.url, endpoint))
                .json(body)
                .send()
                .await?
                .text()
                .await?,
        )
    }

    pub async fn config(&self) -> Result<Config> {
        let components: HashMap<String, serde_json::Value> = self.get("config").await?;
        Ok(Config(
            components
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
        ))
    }

    pub async fn generate_image_from_text(&self, prompt: &str) -> Result<GenerationResult> {
        #[derive(Serialize)]
        struct Request<'a> {
            prompt: &'a str,
        }

        #[derive(Deserialize)]
        struct Response {
            images: Vec<String>,
            info: String,
        }

        let response: Response = self.post("sdapi/v1/txt2img", &Request { prompt }).await?;
        Ok(GenerationResult {
            images: response
                .images
                .iter()
                .map(|b64| Ok(image::load_from_memory(&base64::decode(b64)?)?))
                .collect::<Result<Vec<_>>>()?,
            info: serde_json::from_str(&response.info)?,
        })
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
