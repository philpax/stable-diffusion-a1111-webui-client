use std::collections::HashMap;

use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("invalid url; make sure it starts with http")]
    InvalidUrl,
    #[error("reqwest error")]
    ReqwestError(#[from] reqwest::Error),
    #[error("serde json error")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("Not authenticated")]
    NotAuthenticated,
    #[error("invalid response body (expected {expected:?})")]
    InvalidResponse { expected: String },
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
        value: String,
    },
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
                .json()
                .await?,
        )
    }

    pub async fn config(&self) -> Result<HashMap<u32, ConfigComponent>> {
        let components: HashMap<String, serde_json::Value> = self.get("config").await?;
        Ok(components
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
                            choices: props
                                .get("choices")?
                                .as_array()?
                                .into_iter()
                                .flat_map(|s| Some(s.as_str()?.to_owned()))
                                .collect(),
                            id: props.get("elem_id")?.as_str()?.to_owned(),
                            label: props.get("label")?.as_str()?.to_owned(),
                            value: props.get("value")?.as_str()?.to_owned(),
                        },
                    )),
                    _ => None,
                }
            })
            .collect())
    }

    pub async fn checkpoints(&self) -> Result<Vec<String>> {
        self.config()
            .await?
            .into_values()
            .find_map(|comp| match comp {
                ConfigComponent::Dropdown { id, choices, .. }
                    if id == "setting_sd_model_checkpoint" =>
                {
                    Some(choices)
                }
                _ => None,
            })
            .ok_or_else(|| {
                ClientError::invalid_response("no setting_sd_model_checkpoint component")
            })
    }
}
