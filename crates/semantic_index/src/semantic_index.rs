use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use swiftide::{
    indexing::{
        self, BackoffConfiguration, LanguageModelWithBackOff, Node,
        transformers::{ChunkCode, Embed, MetadataQACode},
    },
    integrations::{
        duckdb::Duckdb, ollama::Ollama, openai::Options, treesitter::SupportedLanguages,
    },
};

const SUPPORTED_LANGUAGES: [SupportedLanguages; 12] = [
    SupportedLanguages::Rust,
    SupportedLanguages::Typescript,
    SupportedLanguages::Python,
    SupportedLanguages::Ruby,
    SupportedLanguages::Javascript,
    SupportedLanguages::Java,
    SupportedLanguages::Go,
    SupportedLanguages::Solidity,
    SupportedLanguages::C,
    SupportedLanguages::Cpp,
    SupportedLanguages::Elixir,
    SupportedLanguages::HTML,
];

pub struct CodebaseIndexer {
    storage_provider: VectorStorageProvider,
    embed_provider: EmbeddingProvider,
    project_dir: PathBuf,
    local_db_path: PathBuf,
}

impl CodebaseIndexer {
    pub fn new(
        embed_provider: EmbeddingProvider,
        storage_provider: VectorStorageProvider,
        project_dir: PathBuf,
        storage_path: PathBuf,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            embed_provider,
            storage_provider,
            project_dir,
            local_db_path: storage_path.join("embeddings.db"),
        })
    }

    pub async fn generate_index(&self, files: &[PathBuf]) -> anyhow::Result<()> {
        let model = match &self.embed_provider {
            EmbeddingProvider::Ollama {
                embed_model,
                prompt_model,
            } => LanguageModelWithBackOff::new(
                Ollama::builder()
                    .default_embed_model(embed_model)
                    .default_prompt_model(prompt_model)
                    // temperature=0 is ideal for embeddings, do not allow the LLM to be creative
                    .default_options(Options::builder().temperature(0.0).build()?)
                    .build()?,
                BackoffConfiguration::default(),
            ),
        };

        let storage = match &self.storage_provider {
            VectorStorageProvider::DuckDb => Duckdb::builder()
                .connection(duckdb::Connection::open(
                    self.local_db_path.join("embeddings.db"),
                )?)
                .table_name(generate_table_name_from_project(&self.project_dir))
                .upsert_vectors(true)
                .build()?,
        };

        let mut remaining_files: HashSet<&PathBuf> = HashSet::from_iter(files.iter());
        for language in SUPPORTED_LANGUAGES {
            let extensions = language.file_extensions();
            let files = remaining_files
                .extract_if(|file| {
                    file.extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| extensions.contains(&ext))
                        .unwrap_or(false)
                })
                .filter_map(|file|
                // Skip any files we can't read, since it's possible they were deleted or renamed.
                // We'll pick them up on the next update.
                fs::read_to_string(file).ok())
                .map(Node::new)
                .collect::<Vec<Node>>();
            indexing::Pipeline::from_stream(files)
                .filter_cached(storage.clone())
                .then(MetadataQACode::new(model.clone()))
                .then_chunk(ChunkCode::try_for_language_and_chunk_size(
                    language,
                    10..2048,
                )?)
                .then_in_batch(Embed::new(model.clone()).with_batch_size(10))
                .then_store_with(storage.clone())
                .run()
                .await?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum EmbeddingProvider {
    Ollama {
        embed_model: String,
        prompt_model: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingProviderPreset {
    OllamaNomic,
}

impl From<EmbeddingProviderPreset> for EmbeddingProvider {
    fn from(preset: EmbeddingProviderPreset) -> Self {
        match preset {
            EmbeddingProviderPreset::OllamaNomic => EmbeddingProvider::Ollama {
                embed_model: "nomic-embed-text".to_string(),
                prompt_model: "qwen2.5-coder:1.5b".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorStorageProvider {
    /// Store embeddings locally in a DuckDB database
    DuckDb,
}

fn generate_table_name_from_project(project_dir: &Path) -> String {
    // To avoid issues related to path lengths and Unicode characters in pathnames,
    // hash the project path and use that hash for the table name.
    let project_hash = Sha256::digest(project_dir.to_string_lossy().as_bytes());
    format!("codebase_{project_hash:x}")
}
