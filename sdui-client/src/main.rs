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

        #[arg(short, long)]
        username: Option<String>,

        #[arg(short, long)]
        password: Option<String>,
    }

    let args = Args::parse();
    let client = client::Client::new(
        &args.url,
        args.username.as_deref().zip(args.password.as_deref()),
    )
    .await?;

    println!("{:#?}", client.checkpoints().await?);

    Ok(())
}
