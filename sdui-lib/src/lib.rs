#![deny(missing_docs)]
//! This is a client for the Automatic1111 stable-diffusion web UI.

use std::{collections::HashMap, future::Future};

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
}
impl Client {
    /// Creates a new `Client` and authenticates to the web UI.
    pub async fn new(url: &str, authentication: Authentication<'_>) -> Result<Self> {
        let mut client = RequestClient::new(url).await?;

        match authentication {
            Authentication::None => {}
            Authentication::ApiAuth(username, password) => {
                client.set_authentication_token(base64::encode(format!("{username}:{password}")));
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

        Ok(Self { client })
    }

    /// Generates an image from the provided `request`, which contains a prompt.
    pub fn generate_from_text(
        &self,
        request: &TextToImageGenerationRequest,
    ) -> impl Future<Output = Result<GenerationResult>> {
        let client = self.client.clone();

        #[derive(Serialize)]
        struct Request {
            batch_size: i32,
            cfg_scale: f32,
            denoising_strength: f32,
            eta: f32,
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
            override_settings: OverrideSettings,
            override_settings_restore_afterwards: bool,

            enable_hr: bool,
            firstphase_height: u32,
            firstphase_width: u32,
        }

        let json_request = {
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
                steps: 20,
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
                sampler_index: Sampler::EulerA.to_string(),
                override_settings: OverrideSettings::default(),
                override_settings_restore_afterwards: false,
            };
            let r = &request;
            let b = &request.base;
            Request {
                batch_size: b.batch_size.map(|i| i as i32).unwrap_or(d.batch_size),
                cfg_scale: b.cfg_scale.unwrap_or(d.cfg_scale),
                denoising_strength: b.denoising_strength.unwrap_or(d.denoising_strength),
                enable_hr: r.enable_hr.unwrap_or(d.enable_hr),
                eta: b.eta.unwrap_or(d.eta),
                firstphase_height: r.firstphase_height.unwrap_or(d.firstphase_height),
                firstphase_width: r.firstphase_width.unwrap_or(d.firstphase_width),
                height: b.height.unwrap_or(d.height),
                n_iter: b.batch_count.unwrap_or(d.n_iter),
                negative_prompt: b.negative_prompt.clone().unwrap_or(d.negative_prompt),
                prompt: b.prompt.to_owned(),
                restore_faces: b.restore_faces.unwrap_or(d.restore_faces),
                s_churn: b.s_churn.unwrap_or(d.s_churn),
                s_noise: b.s_noise.unwrap_or(d.s_noise),
                s_tmax: b.s_tmax.unwrap_or(d.s_tmax),
                s_tmin: b.s_tmin.unwrap_or(d.s_tmin),
                sampler_index: b.sampler.map(|s| s.to_string()).unwrap_or(d.sampler_index),
                seed: b.seed.unwrap_or(d.seed),
                seed_resize_from_h: b
                    .seed_resize_from_h
                    .map(|i| i as i32)
                    .unwrap_or(d.seed_resize_from_h),
                seed_resize_from_w: b
                    .seed_resize_from_w
                    .map(|i| i as i32)
                    .unwrap_or(d.seed_resize_from_w),
                steps: b.steps.unwrap_or(d.steps),
                styles: b.styles.clone().unwrap_or(d.styles),
                subseed: b.subseed.unwrap_or(d.subseed),
                subseed_strength: b.subseed_strength.unwrap_or(d.subseed_strength),
                tiling: b.tiling.unwrap_or(d.tiling),
                width: b.width.unwrap_or(d.width),
                override_settings: OverrideSettings::from_model(r.base.model.as_ref()),
                override_settings_restore_afterwards: false,
            }
        };

        async move {
            let tiling = json_request.tiling;
            Self::issue_generation_task(
                client,
                "sdapi/v1/txt2img".to_string(),
                json_request,
                tiling,
            )
            .await
        }
    }

    /// Generates an image from the provided `request`, which contains both an image and a prompt.
    pub fn generate_from_image_and_text(
        &self,
        request: &ImageToImageGenerationRequest,
    ) -> impl Future<Output = Result<GenerationResult>> {
        let client = self.client.clone();

        #[derive(Serialize)]
        struct Request {
            batch_size: i32,
            cfg_scale: f32,
            denoising_strength: f32,
            eta: f32,
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
            override_settings: OverrideSettings,
            override_settings_restore_afterwards: bool,

            init_images: Vec<String>,
            resize_mode: u32,
            mask: Option<String>,
            mask_blur: u32,
            inpainting_fill: u32,
            inpaint_full_res: bool,
            inpaint_full_res_padding: u32,
            inpainting_mask_invert: u32,
            include_init_images: bool,
        }

        let json_request = (|| {
            let d = Request {
                denoising_strength: 0.0,
                prompt: String::new(),
                styles: vec![],
                seed: -1,
                subseed: -1,
                subseed_strength: 0.0,
                seed_resize_from_h: -1,
                seed_resize_from_w: -1,
                batch_size: 1,
                n_iter: 1,
                steps: 20,
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
                sampler_index: Sampler::EulerA.to_string(),
                override_settings: OverrideSettings::default(),
                override_settings_restore_afterwards: true,

                init_images: vec![],
                resize_mode: 0,
                mask: None,
                mask_blur: 4,
                inpainting_fill: 0,
                inpaint_full_res: true,
                inpaint_full_res_padding: 0,
                inpainting_mask_invert: 0,
                include_init_images: false,
            };
            let r = &request;
            let b = &request.base;
            Ok::<_, ClientError>(Request {
                batch_size: b.batch_size.map(|i| i as i32).unwrap_or(d.batch_size),
                cfg_scale: b.cfg_scale.unwrap_or(d.cfg_scale),
                denoising_strength: b.denoising_strength.unwrap_or(d.denoising_strength),
                eta: b.eta.unwrap_or(d.eta),
                height: b.height.unwrap_or(d.height),
                n_iter: b.batch_count.unwrap_or(d.n_iter),
                negative_prompt: b.negative_prompt.clone().unwrap_or(d.negative_prompt),
                prompt: b.prompt.to_owned(),
                restore_faces: b.restore_faces.unwrap_or(d.restore_faces),
                s_churn: b.s_churn.unwrap_or(d.s_churn),
                s_noise: b.s_noise.unwrap_or(d.s_noise),
                s_tmax: b.s_tmax.unwrap_or(d.s_tmax),
                s_tmin: b.s_tmin.unwrap_or(d.s_tmin),
                sampler_index: b.sampler.map(|s| s.to_string()).unwrap_or(d.sampler_index),
                seed: b.seed.unwrap_or(d.seed),
                seed_resize_from_h: b
                    .seed_resize_from_h
                    .map(|i| i as i32)
                    .unwrap_or(d.seed_resize_from_h),
                seed_resize_from_w: b
                    .seed_resize_from_w
                    .map(|i| i as i32)
                    .unwrap_or(d.seed_resize_from_w),
                steps: b.steps.unwrap_or(d.steps),
                styles: b.styles.clone().unwrap_or(d.styles),
                subseed: b.subseed.unwrap_or(d.subseed),
                subseed_strength: b.subseed_strength.unwrap_or(d.subseed_strength),
                tiling: b.tiling.unwrap_or(d.tiling),
                width: b.width.unwrap_or(d.width),
                override_settings: OverrideSettings::from_model(r.base.model.as_ref()),
                override_settings_restore_afterwards: false,

                init_images: r
                    .images
                    .iter()
                    .map(encode_image_to_base64)
                    .collect::<core::result::Result<Vec<_>, _>>()?,
                resize_mode: r.resize_mode.unwrap_or_default().into(),
                mask: r.mask.as_ref().map(encode_image_to_base64).transpose()?,
                mask_blur: r.mask_blur.unwrap_or(d.mask_blur),
                inpainting_fill: match r.inpainting_fill_mode.unwrap_or_default() {
                    InpaintingFillMode::Fill => 0,
                    InpaintingFillMode::Original => 1,
                    InpaintingFillMode::LatentNoise => 2,
                    InpaintingFillMode::LatentNothing => 3,
                },
                inpaint_full_res: r.inpaint_full_resolution,
                inpaint_full_res_padding: r
                    .inpaint_full_resolution_padding
                    .unwrap_or(d.inpaint_full_res_padding),
                inpainting_mask_invert: r.inpainting_mask_invert as _,
                include_init_images: false,
            })
        })();

        async move {
            let json_request = json_request?;
            let tiling = json_request.tiling;
            Self::issue_generation_task(
                client,
                "sdapi/v1/img2img".to_string(),
                json_request,
                tiling,
            )
            .await
        }
    }

    /// Retrieves the progress of the current generation.
    ///
    /// Note that:
    ///     - this does not disambiguate between generations (the WebUI does not expose details on its queue)
    ///     - this will return 0% if there is no generation underway, which can be confusing after a
    ///       generation finishes
    pub async fn progress(&self) -> Result<GenerationProgress> {
        #[derive(Deserialize)]
        struct State {
            job_timestamp: String,
        }

        #[derive(Deserialize)]
        struct Response {
            eta_relative: f32,
            progress: f32,
            current_image: Option<String>,
            state: Option<State>,
        }

        let response: Response = self.client.get("sdapi/v1/progress").await?;
        Ok(GenerationProgress {
            eta_seconds: response.eta_relative.max(0.0),
            progress_factor: response.progress.clamp(0.0, 1.0),
            current_image: response
                .current_image
                .map(|i| decode_image_from_base64(&i))
                .transpose()?,
            job_timestamp: response
                .state
                .map(|s| chrono::NaiveDateTime::parse_from_str(&s.job_timestamp, "%Y%m%d%H%M%S"))
                .transpose()?
                .map(|dt| dt.and_local_timezone(chrono::Local).unwrap()),
        })
    }

    /// Upscales the given `image` and applies additional (optional) post-processing.
    pub async fn postprocess(
        &self,
        image: &DynamicImage,
        request: &PostprocessRequest,
    ) -> Result<DynamicImage> {
        #[derive(Serialize)]
        struct RequestRaw<'a> {
            image: &'a str,
            resize_mode: u32,
            upscaler_1: &'a str,
            upscaler_2: &'a str,
            upscaling_resize: f32,

            #[serde(skip_serializing_if = "Option::is_none")]
            codeformer_visibility: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            codeformer_weight: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            extras_upscaler_2_visibility: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            gfpgan_visibility: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            upscale_first: Option<bool>,
        }

        #[derive(Deserialize)]
        struct ResponseRaw {
            image: String,
        }

        let response: ResponseRaw = self
            .client
            .post(
                "sdapi/v1/extra-single-image",
                &RequestRaw {
                    image: &encode_image_to_base64(image)?,
                    resize_mode: request.resize_mode.into(),
                    upscaler_1: &request.upscaler_1.to_string(),
                    upscaler_2: &request.upscaler_2.to_string(),
                    upscaling_resize: request.scale_factor,

                    codeformer_visibility: request.codeformer_visibility,
                    codeformer_weight: request.codeformer_weight,
                    extras_upscaler_2_visibility: request.upscaler_2_visibility,
                    gfpgan_visibility: request.gfpgan_visibility,
                    upscale_first: request.upscale_first,
                },
            )
            .await?;

        decode_image_from_base64(&response.image)
    }

    /// Interrogates the given `image` with the `interrogator` to generate a caption.
    pub async fn interrogate(
        &self,
        image: &DynamicImage,
        interrogator: Interrogator,
    ) -> Result<String> {
        #[derive(Serialize)]
        struct RequestRaw<'a> {
            image: &'a str,
            model: &'a str,
        }

        #[derive(Deserialize)]
        struct ResponseRaw {
            caption: String,
        }

        let response: ResponseRaw = self
            .client
            .post(
                "sdapi/v1/interrogate",
                &RequestRaw {
                    image: &encode_image_to_base64(image)?,
                    model: match interrogator {
                        Interrogator::Clip => "clip",
                        Interrogator::DeepDanbooru => "deepdanbooru",
                    },
                },
            )
            .await?;

        Ok(response.caption)
    }

    /// Gets the PNG info for the image (assumed to be valid PNG)
    pub async fn png_info(&self, image_bytes: &[u8]) -> Result<String> {
        #[derive(Serialize)]
        struct RequestRaw<'a> {
            image: &'a str,
        }

        #[derive(Deserialize)]
        struct ResponseRaw {
            info: String,
        }

        let response: ResponseRaw = self
            .client
            .post(
                "sdapi/v1/png-info",
                &RequestRaw {
                    image: &base64::encode(image_bytes),
                },
            )
            .await?;

        Ok(response.info)
    }

    /// Get the embeddings
    pub async fn embeddings(&self) -> Result<Embeddings> {
        #[derive(Deserialize)]
        struct EmbeddingRaw {
            step: Option<u32>,
            sd_checkpoint: Option<String>,
            sd_checkpoint_name: Option<String>,
            shape: u32,
            vectors: u32,
        }

        #[derive(Deserialize)]
        struct ResponseRaw {
            loaded: HashMap<String, EmbeddingRaw>,
            skipped: HashMap<String, EmbeddingRaw>,
        }

        fn convert_embeddings(hm: HashMap<String, EmbeddingRaw>) -> HashMap<String, Embedding> {
            hm.into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        Embedding {
                            step: v.step,
                            sd_checkpoint: v.sd_checkpoint,
                            sd_checkpoint_name: v.sd_checkpoint_name,
                            shape: v.shape,
                            vectors: v.vectors,
                        },
                    )
                })
                .collect()
        }

        let response: ResponseRaw = self.client.get("sdapi/v1/embeddings").await?;

        Ok(Embeddings {
            loaded: convert_embeddings(response.loaded),
            skipped: convert_embeddings(response.skipped),
        })
    }

    /// Get the options
    pub async fn options(&self) -> Result<Options> {
        #[derive(Deserialize)]
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
                    .flat_map(|r| {
                        Upscaler::try_from(r.model_name.as_deref().unwrap_or(r.name.as_str())).ok()
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
impl Client {
    async fn issue_generation_task<R: Serialize + Send + Sync + 'static>(
        client: RequestClient,
        url: String,
        request: R,
        tiling: bool,
    ) -> Result<GenerationResult> {
        #[derive(Deserialize)]
        struct Response {
            images: Vec<String>,
            info: String,
        }

        #[derive(Deserialize)]
        pub struct InfoResponse {
            all_negative_prompts: Vec<String>,
            all_prompts: Vec<String>,

            all_seeds: Vec<i64>,
            seed_resize_from_h: i32,
            seed_resize_from_w: i32,

            all_subseeds: Vec<i64>,
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
            steps: u32,
        }

        let response: Response = client.post(&url, &request).await?;
        let images = response
            .images
            .iter()
            .map(|b64| decode_image_from_base64(b64.as_str()))
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
                tiling,

                cfg_scale: raw.cfg_scale,
                denoising_strength: raw.denoising_strength,
                restore_faces: raw.restore_faces,
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
    }
}

fn decode_image_from_base64(b64: &str) -> Result<DynamicImage> {
    Ok(image::load_from_memory(&base64::decode(b64)?)?)
}

fn encode_image_to_base64(image: &DynamicImage) -> image::ImageResult<String> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    image.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
    Ok(base64::encode(bytes))
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
    /// The current image being generated, if available.
    pub current_image: Option<DynamicImage>,
    /// The timestamp that the current job was started. Can be used to disambiguate between jobs.
    pub job_timestamp: Option<chrono::DateTime<chrono::Local>>,
}
impl GenerationProgress {
    /// Whether or not the generation has completed.
    pub fn is_finished(&self) -> bool {
        self.progress_factor >= 1.0
    }
}

/// The parameters to the generation that are shared between text-to-image synthesis
/// and image-to-image synthesis.
///
/// Consider using the [Default] trait to fill in the
/// parameters that you don't need to fill in.
#[derive(Default, Clone, Debug)]
pub struct BaseGenerationRequest {
    /// The prompt
    pub prompt: String,
    /// The negative prompt (elements to avoid from the generation)
    pub negative_prompt: Option<String>,

    /// The number of images in each batch
    pub batch_size: Option<u32>,
    /// The number of batches
    pub batch_count: Option<u32>,

    /// The width of the generated image
    pub width: Option<u32>,
    /// The height of the generated image
    pub height: Option<u32>,

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
    pub model: Option<Model>,

    /// Whether or not the image should be tiled at the edges
    pub tiling: Option<bool>,
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

/// Parameters for a text-to-image generation.
///
/// Consider using the [Default] trait to fill in the
/// parameters that you don't need to fill in.
#[derive(Default, Clone, Debug)]
pub struct TextToImageGenerationRequest {
    /// The base parameters for this generation request.
    pub base: BaseGenerationRequest,

    /// The width of the first phase of the generated image
    pub firstphase_width: Option<u32>,
    /// The height of the first phase of the generated image
    pub firstphase_height: Option<u32>,

    /// Unknown
    pub enable_hr: Option<bool>,
}

/// Parameters for an image-to-image generation.
///
/// Consider using the [Default] trait to fill in the
/// parameters that you don't need to fill in.
#[derive(Default, Clone, Debug)]
pub struct ImageToImageGenerationRequest {
    /// The base parameters for this generation request.
    pub base: BaseGenerationRequest,

    /// The images to alter.
    pub images: Vec<DynamicImage>,

    /// How the image will be resized to match the generation resolution
    pub resize_mode: Option<ResizeMode>,

    /// The mask to apply
    pub mask: Option<DynamicImage>,

    /// The amount to blur the mask
    pub mask_blur: Option<u32>,

    /// How the area to be inpainted will be initialized
    pub inpainting_fill_mode: Option<InpaintingFillMode>,

    /// Whether or not to inpaint at full resolution
    pub inpaint_full_resolution: bool,

    /// The amount of padding to apply to the full-resolution padding
    pub inpaint_full_resolution_padding: Option<u32>,

    /// By default, the masked area is inpainted. If this is turned on, the unmasked area
    /// will be inpainted.
    pub inpainting_mask_invert: bool,
}

/// The result of the generation.
pub struct GenerationResult {
    /// The images produced by the generator.
    pub images: Vec<DynamicImage>,
    /// The information associated with this generation.
    pub info: GenerationInfo,
}

/// The information associated with a generation.
#[derive(Debug, Clone)]
pub struct GenerationInfo {
    /// The prompts used for each image in the generation.
    pub prompts: Vec<String>,
    /// The negative prompt for each image in the generation.
    pub negative_prompts: Vec<String>,
    /// The seeds for the images; each seed corresponds to an image.
    pub seeds: Vec<i64>,
    /// The subseeds for the images; each seed corresponds to an image.
    pub subseeds: Vec<i64>,
    /// The strength of the subseed.
    pub subseed_strength: f32,
    /// The width of the generated images.
    pub width: u32,
    /// The height of the generated images.
    pub height: u32,
    /// The sampler that was used for this generation.
    pub sampler: Sampler,
    /// The number of steps that were used for each generation.
    pub steps: u32,
    /// Whether or not the image should be tiled at the edges
    pub tiling: bool,

    /// The Classifier-Free Guidance scale; how strongly the prompt was
    /// applied to the generation
    pub cfg_scale: f32,
    /// The denoising strength
    pub denoising_strength: f32,

    /// Whether or not the face restoration was applied
    pub restore_faces: bool,

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

/// A request to post-process an image. See [Client::postprocess].
#[derive(Default)]
pub struct PostprocessRequest {
    /// How the image should be resized to fit the target resolution if the
    /// resolution doesn't fit within the frame
    pub resize_mode: ResizeMode,
    /// The first upscaler to use
    pub upscaler_1: Upscaler,
    /// The second upscaler to use
    pub upscaler_2: Upscaler,
    /// The scale factor to use
    pub scale_factor: f32,

    /// How much of CodeFormer's result is blended into the result? [0-1]
    pub codeformer_visibility: Option<f32>,
    /// How strong is CodeFormer's effect? [0-1]
    pub codeformer_weight: Option<f32>,
    /// How much of the second upscaler's result is blended into the result? [0-1]
    pub upscaler_2_visibility: Option<f32>,
    /// How much of GFPGAN's result is blended into the result? [0-1]
    pub gfpgan_visibility: Option<f32>,
    /// Should upscaling occur before face restoration?
    pub upscale_first: Option<bool>,
}

macro_rules! define_user_friendly_enum {
    ($enum_name:ident, $doc:literal, {$(($name:ident, $friendly_name:literal)),*}) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
        pub enum $enum_name {
            $(
                #[doc = $friendly_name]
                $name
            ),*
        }
        impl std::fmt::Display for $enum_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", match self {
                    $(
                        Self::$name => $friendly_name
                    ),*
                })
            }
        }
        impl std::fmt::Debug for $enum_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(self, f)
            }
        }
        impl TryFrom<&str> for $enum_name {
            type Error = ();

            fn try_from(s: &str) -> core::result::Result<Self, ()> {
                match s {
                    $(
                        $friendly_name => Ok(Self::$name),
                    )*
                    _ => Err(()),
                }
            }
        }
        impl $enum_name {
            /// All of the possible values.
            pub const VALUES: &[Self] = &[
                $(Self::$name),*
            ];
        }
    }
}

define_user_friendly_enum!(
    Sampler,
    "The sampler to use for the generation.",
    {
        (EulerA, "Euler a"),
        (Euler, "Euler"),
        (Lms, "LMS"),
        (Heun, "Heun"),
        (Dpm2, "DPM2"),
        (Dpm2A, "DPM2 a"),
        (DpmPP2SA, "DPM++ 2S a"),
        (DpmPP2M, "DPM++ 2M"),
        (DpmPPSDE, "DPM++ SDE"),
        (DpmFast, "DPM fast"),
        (DpmAdaptive, "DPM adaptive"),
        (LmsKarras, "LMS Karras"),
        (Dpm2Karras, "DPM2 Karras"),
        (Dpm2AKarras, "DPM2 a Karras"),
        (DpmPP2SAKarras, "DPM++ 2S a Karras"),
        (DpmPP2MKarras, "DPM++ 2M Karras"),
        (DpmPPSDEKarras, "DPM++ SDE Karras"),
        (Ddim, "DDIM"),
        (Plms, "PLMS")
    }
);

define_user_friendly_enum!(
    Interrogator,
    "Supported interrogators for [Client::interrogate]",
    {
        (Clip, "CLIP"),
        (DeepDanbooru, "DeepDanbooru")
    }
);

define_user_friendly_enum!(
    ResizeMode,
    "How to resize the image for image-to-image generation",
    {
        (Resize, "Just resize"),
        (CropAndResize, "Crop and resize"),
        (ResizeAndFill, "Resize and fill")
    }
);
impl Default for ResizeMode {
    fn default() -> Self {
        Self::Resize
    }
}
impl From<ResizeMode> for u32 {
    fn from(mode: ResizeMode) -> Self {
        match mode {
            ResizeMode::Resize => 0,
            ResizeMode::CropAndResize => 1,
            ResizeMode::ResizeAndFill => 2,
        }
    }
}

define_user_friendly_enum!(
    InpaintingFillMode,
    "How the area to be inpainted will be initialized",
    {
        (Fill, "Fill"),
        (Original, "Original"),
        (LatentNoise, "Latent noise"),
        (LatentNothing, "Latent nothing")
    }
);
impl Default for InpaintingFillMode {
    fn default() -> Self {
        Self::Original
    }
}

define_user_friendly_enum!(
    Upscaler,
    "Upscaler",
    {
        (None, "None"),
        (Lanczos, "Lanczos"),
        (Nearest, "Nearest"),
        (Ldsr, "LDSR"),
        (ScuNetGan, "ScuNET GAN"),
        (ScuNetPSNR, "ScuNET PSNR"),
        (SwinIR4x, "SwinIR 4x"),
        (ESRGAN4x, "ESRGAN_4x")
    }
);
impl Default for Upscaler {
    fn default() -> Self {
        Self::None
    }
}

/// The currently set options for the UI
#[derive(Debug, Clone)]
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

/// Model
#[derive(Debug, Clone)]
pub struct Model {
    /// Title of the model
    pub title: String,
    /// Name of the model
    pub name: String,
}

/// Prompt style
#[derive(Debug, Clone)]
pub struct PromptStyle {
    /// Name of the style
    pub name: String,
    /// Prompt of the style
    pub prompt: Option<String>,
    /// Negative prompt of the style
    pub negative_prompt: Option<String>,
}

/// Artist
#[derive(Debug, Clone)]
pub struct Artist {
    /// Name of the artist
    pub name: String,
    /// Category the artist belongs to
    pub category: String,
}

/// A textual inversion embedding
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The number of steps that were used to train this embedding, if available
    pub step: Option<u32>,
    /// The hash of the checkpoint this embedding was trained on, if available
    pub sd_checkpoint: Option<String>,
    /// The name of the checkpoint this embedding was trained on, if available
    ///
    /// Note that this is the name that was used by the trainer; for a stable identifier, use [sd_checkpoint] instead
    pub sd_checkpoint_name: Option<String>,
    /// The length of each individual vector in the embedding
    pub shape: u32,
    /// The number of vectors in the embedding
    pub vectors: u32,
}

/// All available textual inversion embeddings
#[derive(Debug, Clone)]
pub struct Embeddings {
    /// Embeddings loaded for the current model
    pub loaded: HashMap<String, Embedding>,
    /// Embeddings skipped for the current model (likely due to architecture incompatibility)
    pub skipped: HashMap<String, Embedding>,
}
impl Embeddings {
    /// Iterator over all embeddings available, including currently unloaded ones
    pub fn all(&self) -> impl Iterator<Item = (&String, &Embedding)> {
        self.loaded.iter().chain(self.skipped.iter())
    }
}

#[allow(dead_code)]
mod config {
    use crate::{ClientError, RequestClient, Result};
    use std::collections::HashMap;

    #[derive(Debug)]
    enum ConfigComponent {
        Dropdown { choices: Vec<String>, id: String },
        Radio { choices: Vec<String>, id: String },
    }

    /// The top-level Gradio configuration for the Web UI.
    ///
    /// Used to get things that aren't in the API.
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

    fn extract_string_array(value: &serde_json::Value) -> Option<Vec<String>> {
        Some(
            value
                .as_array()?
                .iter()
                .flat_map(|s| Some(s.as_str()?.to_owned()))
                .collect(),
        )
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

#[derive(Serialize, Deserialize, Default)]
struct OverrideSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    sd_model_checkpoint: Option<String>,
}
impl OverrideSettings {
    fn from_model(model: Option<&Model>) -> Self {
        Self {
            sd_model_checkpoint: model.map(|m| m.title.clone()),
        }
    }
}
