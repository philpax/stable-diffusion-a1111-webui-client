use clap::{Parser, ValueEnum};
use stable_diffusion_a1111_webui_client as client;
use std::{
    fmt::Debug,
    io::{BufRead, Write},
    iter::IntoIterator,
    path::PathBuf,
    time::Duration,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    macro_rules! remap_enum {
        ($enum_name:ident, {$($name:ident,)*}) => {
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
            pub enum $enum_name {
                $(
                    $name
                ),*
            }

            impl From<$enum_name> for client::$enum_name {
                fn from(s: $enum_name) -> Self {
                    match s {
                        $($enum_name::$name => client::$enum_name::$name,)*
                    }
                }
            }
        }
    }

    remap_enum!(Sampler, {
        EulerA,
        Euler,
        Lms,
        Heun,
        Dpm2,
        Dpm2A,
        DpmPP2SA,
        DpmPP2M,
        DpmPPSDE,
        DpmFast,
        DpmAdaptive,
        LmsKarras,
        Dpm2Karras,
        Dpm2AKarras,
        DpmPP2SAKarras,
        DpmPP2MKarras,
        DpmPPSDEKarras,
        Ddim,
        Plms,
    });

    remap_enum!(Interrogator, {
        Clip,
        DeepDanbooru,
    });

    remap_enum!(Upscaler, {
        None,
        Lanczos,
        Nearest,
        Ldsr,
        ScuNetGan,
        ScuNetPSNR,
        SwinIR4x,
        ESRGAN4x,
    });

    /// Client for Automatic1111's Stable Diffusion web UI
    #[derive(Parser)]
    #[command(author, version, about, long_about = None)]
    struct Args {
        /// The URL of the server to connect to
        #[arg()]
        url: String,

        /// The username to use for authentication. Must also pass in `password`.
        #[arg(short, long)]
        username: Option<String>,

        /// The password to use for authentication. Must also pass in `username`.
        #[arg(short, long)]
        password: Option<String>,
    }

    #[derive(Parser)]
    #[command(author, version, about, long_about = None)]
    #[command(propagate_version = true)]
    enum Command {
        Exit,
        /// Generates an image from text
        GenerateFromText {
            /// The prompt to generate
            #[arg()]
            prompt: String,

            /// The number of images to produce
            #[arg(short, long)]
            count: Option<u32>,

            /// The number of denoising steps
            #[arg(long)]
            steps: Option<u32>,

            /// The width of the image
            #[arg(long)]
            width: Option<u32>,

            /// The height of the image
            #[arg(long)]
            height: Option<u32>,

            /// The sampler to use
            #[arg(long)]
            sampler: Option<Sampler>,

            /// Index of the model to use (from `models`) if desired
            #[arg(long)]
            model: Option<usize>,
        },
        /// Generates an image from image and text
        GenerateFromImageAndText {
            /// The prompt to generate
            #[arg()]
            prompt: String,

            /// The image to use
            #[arg()]
            image: PathBuf,

            /// The denoising strength to apply (between 0 and 1)
            #[arg()]
            denoising_strength: f32,

            /// The number of images to produce
            #[arg(short, long)]
            count: Option<u32>,

            /// The number of denoising steps
            #[arg(long)]
            steps: Option<u32>,

            /// The width of the image
            #[arg(long)]
            width: Option<u32>,

            /// The height of the image
            #[arg(long)]
            height: Option<u32>,

            /// The sampler to use
            #[arg(long)]
            sampler: Option<Sampler>,

            /// Index of the model to use (from `models`) if desired
            #[arg(long)]
            model: Option<usize>,
        },
        /// Postprocesses the given image
        Postprocess {
            /// The image to use
            #[arg()]
            image: PathBuf,

            /// The first upscaler to use
            #[arg(long)]
            upscaler_1: Upscaler,

            /// The second upscaler to use
            #[arg(long)]
            upscaler_2: Upscaler,

            /// The scale factor to use
            #[arg(long)]
            scale_factor: f32,

            /// How much of CodeFormer's result is blended into the result? [0-1]
            #[arg(long)]
            codeformer_visibility: Option<f32>,

            /// How strong is CodeFormer's effect? [0-1]
            #[arg(long)]
            codeformer_weight: Option<f32>,
            /// How much of the second upscaler's result is blended into the result? [0-1]
            #[arg(long)]
            upscaler_2_visibility: Option<f32>,

            /// How much of GFPGAN's result is blended into the result? [0-1]
            #[arg(long)]
            gfpgan_visibility: Option<f32>,

            /// Should upscaling occur before face restoration?
            #[arg(long)]
            upscale_first: Option<bool>,
        },
        /// Interrogates the given image with the specified model
        Interrogate {
            /// The image to interrogate
            #[arg()]
            image: PathBuf,

            /// The interrogator to use
            #[arg()]
            interrogator: Interrogator,
        },
        /// Prints the PNG info for the given image, if available
        PngInfo {
            /// The PNG to read
            #[arg()]
            image: PathBuf,
        },
        Embeddings,
        Options,
        Samplers,
        Upscalers,
        Models,
        Hypernetworks,
        FaceRestorers,
        RealEsrganModels,
        PromptStyles,
        ArtistCategories,
        Artists,
    }

    let args = Args::parse();
    let client = client::Client::new(
        &args.url,
        args.username
            .as_deref()
            .zip(args.password.as_deref())
            .map(|(u, p)| client::Authentication::ApiAuth(u, p))
            .unwrap_or(client::Authentication::None),
    )
    .await?;
    let models = client.models().await?;

    'exit: loop {
        let line = prompt_for_input()?;
        // chain an empty string so that clap ignores arg0
        let parse_result = Command::try_parse_from(
            std::iter::once("".to_string()).chain(shlex::Shlex::new(&line)),
        );
        let cmd = match parse_result {
            Ok(r) => r,
            Err(err) => {
                println!("parse error: {}", err);
                continue;
            }
        };
        match cmd {
            Command::Exit => {
                break 'exit;
            }
            Command::GenerateFromText {
                prompt,
                count,
                steps,
                width,
                height,
                sampler,
                model,
            } => {
                let model = get_model_by_index(&models, model).cloned();
                let task = tokio::task::spawn(client.generate_from_text(
                    &client::TextToImageGenerationRequest {
                        base: client::BaseGenerationRequest {
                            prompt,
                            batch_count: count,
                            steps,
                            width,
                            height,
                            sampler: sampler.map(|s| s.into()),
                            model,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                ));
                save_generation_result(&client, task).await?;
            }
            Command::GenerateFromImageAndText {
                prompt,
                image,
                denoising_strength,
                count,
                steps,
                width,
                height,
                sampler,
                model,
            } => {
                let model = get_model_by_index(&models, model).cloned();
                let image = image::open(image)?;
                let task = tokio::task::spawn(client.generate_from_image_and_text(
                    &client::ImageToImageGenerationRequest {
                        base: client::BaseGenerationRequest {
                            prompt,
                            batch_count: count,
                            steps,
                            width,
                            height,
                            sampler: sampler.map(|s| s.into()),
                            model,
                            denoising_strength: Some(denoising_strength),
                            ..Default::default()
                        },
                        images: vec![image],
                        ..Default::default()
                    },
                ));
                save_generation_result(&client, task).await?;
            }
            Command::Postprocess {
                image,
                upscaler_1,
                upscaler_2,
                scale_factor,
                codeformer_visibility,
                codeformer_weight,
                upscaler_2_visibility,
                gfpgan_visibility,
                upscale_first,
            } => {
                let image = image::open(image)?;

                let result = client
                    .postprocess(
                        &image,
                        &client::PostprocessRequest {
                            resize_mode: client::ResizeMode::Resize,
                            upscaler_1: upscaler_1.into(),
                            upscaler_2: upscaler_2.into(),
                            scale_factor,
                            codeformer_visibility,
                            codeformer_weight,
                            upscaler_2_visibility,
                            gfpgan_visibility,
                            upscale_first,
                        },
                    )
                    .await?;

                result.save("output_postprocess.png")?;
            }
            Command::Interrogate {
                image,
                interrogator,
            } => {
                let image = image::open(image)?;
                let result = client.interrogate(&image, interrogator.into()).await?;
                println!("result: {}", result);
            }
            Command::PngInfo { image } => {
                let result = client.png_info(&std::fs::read(image)?).await?;
                println!("result: {}", result);
            }
            Command::Embeddings => {
                let embeddings = client.embeddings().await?;
                list_unordered_print("Loaded", embeddings.loaded);
                list_unordered_print("Skipped", embeddings.skipped);
            }
            Command::Options => println!("Options: {:?}", client.options().await?),
            Command::Samplers => list_unordered_print("Samplers", client.samplers().await?),
            Command::Upscalers => list_unordered_print("Upscalers", client.upscalers().await?),
            Command::Models => list_ordered_print("Models", client.models().await?.iter()),
            Command::Hypernetworks => {
                list_unordered_print("Hypernetworks", client.hypernetworks().await?)
            }
            Command::FaceRestorers => {
                list_unordered_print("FaceRestorers", client.face_restorers().await?)
            }
            Command::RealEsrganModels => {
                list_unordered_print("RealEsrganModels", client.realesrgan_models().await?)
            }
            Command::PromptStyles => {
                list_unordered_print("PromptStyles", client.prompt_styles().await?)
            }
            Command::ArtistCategories => {
                list_unordered_print("ArtistCategories", client.artist_categories().await?)
            }
            Command::Artists => list_unordered_print("Artists", client.artists().await?),
        }
    }

    Ok(())
}

fn prompt_for_input() -> anyhow::Result<String> {
    print!("> ");
    std::io::stdout().lock().flush()?;

    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_owned())
}

fn get_model_by_index(models: &[client::Model], index: Option<usize>) -> Option<&client::Model> {
    if let Some(index) = index {
        let model = models.get(index);
        if model.is_none() {
            println!("warn: specified model is invalid, using set model");
        }
        model
    } else {
        None
    }
}

async fn save_generation_result(
    client: &client::Client,
    task: tokio::task::JoinHandle<client::Result<client::GenerationResult>>,
) -> anyhow::Result<()> {
    loop {
        let progress = client.progress().await?;
        println!(
            "{:.02}% complete, {} seconds remaining (job started at {})",
            progress.progress_factor * 100.0,
            progress.eta_seconds,
            progress
                .job_timestamp
                .map(|t| t.to_rfc3339())
                .unwrap_or_else(|| "unknown time".to_string())
        );
        tokio::time::sleep(Duration::from_millis(250)).await;

        if progress.is_finished() || task.is_finished() {
            break;
        }
    }
    let result = task.await??;
    println!("info: {:?}", result.info);
    for (i, image) in result.pngs.into_iter().enumerate() {
        std::fs::write(format!("output_{i}.png"), image)?;
    }

    Ok(())
}

fn list_unordered_print<D: Debug>(title: &str, values: impl IntoIterator<Item = D>) {
    println!("{title}");
    for value in values.into_iter() {
        println!("- {value:?}");
    }
}

fn list_ordered_print<D: Debug>(title: &str, values: impl Iterator<Item = D>) {
    println!("{title}");
    for (index, value) in values.enumerate() {
        println!("{index}: {value:?}");
    }
}
