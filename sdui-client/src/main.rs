use clap::Parser;
use stable_diffusion_a1111_webui_client as client;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
    }

    let args = Args::parse();
    let client = client::Client::new(
        &args.url,
        args.username.as_deref().zip(args.password.as_deref()),
    )
    .await?;

    let config = client.config().await?;
    println!("checkpoints: {:?}", config.checkpoints()?);
    println!("embeddings: {:?}", config.embeddings()?);
    println!("hypernetwork: {:?}", config.hypernetwork()?);
    println!("txt2img_samplers: {:?}", config.txt2img_samplers()?);

    let result = client.generate_image_from_text(&args.prompt).await?;
    println!("info: {:?}", result.info);
    for (i, image) in result.images.into_iter().enumerate() {
        image.save(format!("output_{i}.png"))?;
    }

    Ok(())
}
