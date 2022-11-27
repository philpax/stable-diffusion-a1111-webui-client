use clap::{Parser, ValueEnum};
use stable_diffusion_a1111_webui_client as client;
use std::{
    fmt::Debug,
    io::{BufRead, Write},
    iter::IntoIterator,
    time::Duration,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
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
    impl From<Sampler> for client::Sampler {
        fn from(s: Sampler) -> Self {
            match s {
                Sampler::EulerA => client::Sampler::EulerA,
                Sampler::Euler => client::Sampler::Euler,
                Sampler::Lms => client::Sampler::Lms,
                Sampler::Heun => client::Sampler::Heun,
                Sampler::Dpm2 => client::Sampler::Dpm2,
                Sampler::Dpm2A => client::Sampler::Dpm2A,
                Sampler::DpmFast => client::Sampler::DpmFast,
                Sampler::DpmAdaptive => client::Sampler::DpmAdaptive,
                Sampler::LmsKarras => client::Sampler::LmsKarras,
                Sampler::Dpm2Karras => client::Sampler::Dpm2Karras,
                Sampler::Dpm2AKarras => client::Sampler::Dpm2AKarras,
                Sampler::Ddim => client::Sampler::Ddim,
                Sampler::Plms => client::Sampler::Plms,
            }
        }
    }

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

    let stdin = std::io::stdin();
    'exit: loop {
        print!("> ");
        std::io::stdout().lock().flush()?;

        let mut line = String::new();
        stdin.lock().read_line(&mut line)?;
        let line = line.trim();
        // chain an empty string so that clap ignores arg0
        let parse_result =
            Command::try_parse_from(std::iter::once("".to_string()).chain(shlex::Shlex::new(line)));
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
            } => {
                let task = client.generate_image_from_text(&client::GenerationRequest {
                    prompt: &prompt,
                    batch_count: count,
                    steps,
                    width,
                    height,
                    sampler: sampler.map(|s| s.into()),
                    ..Default::default()
                });
                loop {
                    let progress = task.progress().await?;
                    println!(
                        "{:.02}% complete, {} seconds remaining",
                        progress.progress_factor * 100.0,
                        progress.eta_seconds
                    );
                    tokio::time::sleep(Duration::from_millis(250)).await;

                    if progress.is_finished() {
                        break;
                    }
                }
                let result = task.await?;
                println!("info: {:?}", result.info);
                for (i, image) in result.images.into_iter().enumerate() {
                    image.save(format!("output_{i}.png"))?;
                }
            }
            Command::Embeddings => list_print("Embeddings", client.embeddings().await?),
            Command::Options => println!("Options: {:?}", client.options().await?),
            Command::Samplers => list_print("Samplers", client.samplers().await?),
            Command::Upscalers => list_print("Upscalers", client.upscalers().await?),
            Command::Models => list_print("Models", client.models().await?),
            Command::Hypernetworks => list_print("Hypernetworks", client.hypernetworks().await?),
            Command::FaceRestorers => list_print("FaceRestorers", client.face_restorers().await?),
            Command::RealEsrganModels => {
                list_print("RealEsrganModels", client.realesrgan_models().await?)
            }
            Command::PromptStyles => list_print("PromptStyles", client.prompt_styles().await?),
            Command::ArtistCategories => {
                list_print("ArtistCategories", client.artist_categories().await?)
            }
            Command::Artists => list_print("Artists", client.artists().await?),
        }
    }

    Ok(())
}

fn list_print<D: Debug>(title: &str, values: impl IntoIterator<Item = D>) {
    println!("{title}");
    for value in values.into_iter() {
        println!("- {value:?}");
    }
}
