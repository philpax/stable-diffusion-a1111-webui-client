use clap::{Parser, ValueEnum};
use stable_diffusion_a1111_webui_client as client;
use std::time::Duration;

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

        /// The prompt to generate
        #[arg()]
        prompt: String,

        /// The username to use for authentication. Must also pass in `password`.
        #[arg(short, long)]
        username: Option<String>,

        /// The password to use for authentication. Must also pass in `username`.
        #[arg(short, long)]
        password: Option<String>,

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

        /// Print information about the client
        #[arg(long)]
        print_information: bool,
    }

    let args = Args::parse();
    let client = client::Client::new(
        &args.url,
        args.username.as_deref().zip(args.password.as_deref()),
    )
    .await?;

    if args.print_information {
        println!("embeddings: {:?}", client.embeddings().await?);
        println!("options: {:?}", client.options().await?);
        println!("samplers: {:?}", client.samplers().await?);
        println!("upscalers: {:?}", client.upscalers().await?);
        println!("models: {:?}", client.models().await?);
        println!("hypernetworks: {:?}", client.hypernetworks().await?);
        println!("face_restorers: {:?}", client.face_restorers().await?);
        println!("realesrgan_models: {:?}", client.realesrgan_models().await?);
        println!("prompt_styles: {:?}", client.prompt_styles().await?);
        println!("artist_categories: {:?}", client.artist_categories().await?);
        println!("artists: {:?}", client.artists().await?);
    }

    let task = client.generate_image_from_text(&client::GenerationRequest {
        prompt: &args.prompt,
        batch_count: args.count,
        steps: args.steps,
        width: args.width,
        height: args.height,
        sampler: args.sampler.map(|s| s.into()),
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

    Ok(())
}
