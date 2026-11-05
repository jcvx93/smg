//! Generate preparation stage: Resolve input, tokenize, create stop decoder

use std::sync::Arc;

use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{common::InputIds, generate::GenerateRequest};
use tracing::error;

use crate::{
    routers::{
        error,
        grpc::{
            context::{PreparationOutput, RequestContext},
            utils::{self, UnknownTokenId},
        },
    },
    worker::WorkerRegistry,
};

/// Generate preparation stage
///
/// Extracts generate-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct GeneratePreparationStage;

impl GeneratePreparationStage {
    /// The stage's work. The chat+generate dispatcher calls this directly, so
    /// the stage's future lives inline in the dispatcher's: one box per stage
    /// call, not a second one for a boxed trait method.
    pub(crate) async fn prepare(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let request = ctx.generate_request_arc();
        self.prepare_generate(ctx, &request).await?;
        Ok(())
    }
}

impl GeneratePreparationStage {
    async fn prepare_generate(
        &self,
        ctx: &mut RequestContext,
        request: &GenerateRequest,
    ) -> Result<(), Response> {
        // Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer = utils::resolve_tokenizer(ctx, "GeneratePreparationStage::prepare_generate")
            .map_err(|e| *e)?;

        let (original_text, token_ids) = match self
            .resolve_generate_input(
                request,
                &tokenizer,
                &ctx.components.worker_registry,
                &ctx.input.model_id,
            )
            .await
        {
            Ok(res) => res,
            Err(msg) => {
                error!(function = "GeneratePreparationStage::execute", error = %msg, "Failed to resolve generate input");
                return Err(error::bad_request("resolve_input_failed", msg));
            }
        };

        // Create stop sequence decoder for generate requests
        let params = request.sampling_params.as_ref();
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            params.and_then(|p| p.stop.as_ref()),
            params.and_then(|p| p.stop_token_ids.as_ref()),
            params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
            params.and_then(|p| p.no_stop_trim).unwrap_or(false),
            params.and_then(|p| p.ignore_eos).unwrap_or(false),
        );

        ctx.state.preparation = Some(PreparationOutput::Generate {
            original_text,
            token_ids,
        });

        // Store stop decoder
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }

    async fn resolve_generate_input(
        &self,
        request: &GenerateRequest,
        tokenizer: &Arc<dyn Tokenizer>,
        worker_registry: &WorkerRegistry,
        model_id: &str,
    ) -> Result<(Option<String>, Vec<u32>), String> {
        if let Some(text) = &request.text {
            return self
                .tokenize_single_text(tokenizer, text)
                .await
                .map(|(original, ids)| (Some(original), ids));
        }

        // Handle input_ids - validate and convert
        if let Some(input_ids) = &request.input_ids {
            return match input_ids {
                InputIds::Single(ids) => {
                    let converted = ids
                        .iter()
                        .map(|&id| u32::try_from(id))
                        .collect::<Result<Vec<u32>, _>>()
                        .map_err(|_| "input_ids must be non-negative".to_string())?;
                    reject_unknown_token_id(tokenizer.as_ref(), &converted, || {
                        utils::reported_embedding_rows(worker_registry, model_id)
                    })?;
                    Ok((None, converted))
                }
                InputIds::Batch(_) => {
                    Err("Batch input_ids are not supported over gRPC generate yet".to_string())
                }
            };
        }

        Err("Either `text` or `input_ids` must be provided".to_string())
    }

    async fn tokenize_single_text(
        &self,
        tokenizer: &Arc<dyn Tokenizer>,
        text: &str,
    ) -> Result<(String, Vec<u32>), String> {
        // Don't add special tokens - raw text generation uses text as-is
        let encoding = utils::encode_blocking(tokenizer.clone(), text.to_string(), false)
            .await
            .map_err(|e| format!("Tokenization failed: {e}"))?;
        Ok((text.to_string(), encoding.into_token_ids()))
    }
}

/// Refuses the first id that can reach no row of the engine's embedding table, naming its
/// position, its value and the bound it missed; the rule is `utils::first_unknown_token_id`,
/// shared with the `logit_bias` keys.
fn reject_unknown_token_id(
    tokenizer: &dyn Tokenizer,
    ids: &[u32],
    embedding_rows: impl FnOnce() -> Option<usize>,
) -> Result<(), String> {
    match utils::first_unknown_token_id(tokenizer, ids, embedding_rows) {
        None => Ok(()),
        Some(UnknownTokenId { index, id, bound }) => Err(format!(
            "input_ids[{index}] = {id} is not a token of this model: {bound}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::{
        traits::{Decoder, Encoder, Encoding, SpecialTokens},
        MockTokenizer,
    };

    use super::*;

    /// A tokenizer that reports no vocabulary; every other answer is the mock's.
    struct NoVocabulary(MockTokenizer);

    impl Encoder for NoVocabulary {
        fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
            self.0.encode(input, add_special_tokens)
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            self.0.encode_batch(inputs, add_special_tokens)
        }
    }

    impl Decoder for NoVocabulary {
        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
            self.0.decode(token_ids, skip_special_tokens)
        }
    }

    impl Tokenizer for NoVocabulary {
        fn vocab_size(&self) -> usize {
            0
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            self.0.get_special_tokens()
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.0.token_to_id(token)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.0.id_to_token(id)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// The mock's base vocabulary is its token count; its added tokens sit far above it.
    fn tokenizer() -> MockTokenizer {
        MockTokenizer::new().with_tokens(&[("<|added|>", 50_000)])
    }

    #[test]
    fn an_id_inside_the_base_vocabulary_passes_without_a_lookup() {
        let tokenizer = tokenizer();
        let inside: Vec<u32> = (0..tokenizer.vocab_size() as u32).collect();
        assert_eq!(
            reject_unknown_token_id(&tokenizer, &inside, || None),
            Ok(())
        );
    }

    #[test]
    fn an_added_token_above_the_base_vocabulary_passes_by_lookup() {
        let tokenizer = tokenizer();
        assert!(50_000 >= tokenizer.vocab_size() as u32);
        assert_eq!(
            reject_unknown_token_id(&tokenizer, &[1, 50_000, 2], || None),
            Ok(())
        );
    }

    #[test]
    fn the_first_unknown_id_is_named_with_its_position_and_the_bound() {
        let tokenizer = tokenizer();
        let base = tokenizer.vocab_size();
        let unknown = base as u32 + 7;
        let error = reject_unknown_token_id(&tokenizer, &[1, 2, unknown, unknown + 1], || None)
            .expect_err("an unknown id is refused");
        assert!(
            error.starts_with(&format!("input_ids[2] = {unknown} ")),
            "{error}"
        );
        assert!(error.contains(&format!("holds {base} tokens")), "{error}");
    }

    #[test]
    fn an_id_the_tokenizer_does_not_know_passes_inside_the_reported_table() {
        let tokenizer = tokenizer();
        let gap = tokenizer.vocab_size() as u32 + 7;
        assert_eq!(
            reject_unknown_token_id(&tokenizer, &[1, gap, 50_000], || Some(2_000)),
            Ok(())
        );
    }

    #[test]
    fn an_id_past_the_reported_table_is_refused_naming_the_table() {
        let tokenizer = tokenizer();
        let error = reject_unknown_token_id(&tokenizer, &[1, 1_999, 2_000], || Some(2_000))
            .expect_err("an id past the table is refused");
        assert!(error.starts_with("input_ids[2] = 2000 "), "{error}");
        assert!(error.contains("embedding table holds 2000 rows"), "{error}");
    }

    #[test]
    fn the_table_is_not_read_while_the_tokenizer_knows_every_id() {
        let tokenizer = tokenizer();
        let inside: Vec<u32> = (0..tokenizer.vocab_size() as u32).chain([50_000]).collect();
        let result = reject_unknown_token_id(&tokenizer, &inside, || {
            panic!("the table is read only after an unknown id")
        });
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn a_tokenizer_without_a_vocabulary_is_bounded_by_the_reported_table() {
        let tokenizer = NoVocabulary(MockTokenizer::new());
        assert_eq!(
            reject_unknown_token_id(&tokenizer, &[5, 1_999], || Some(2_000)),
            Ok(())
        );
        let error = reject_unknown_token_id(&tokenizer, &[5, 2_000], || Some(2_000))
            .expect_err("past the table");
        assert!(error.starts_with("input_ids[1] = 2000 "), "{error}");
    }

    #[test]
    fn a_tokenizer_without_a_vocabulary_skips_the_bound() {
        let tokenizer = NoVocabulary(MockTokenizer::new());
        assert_eq!(
            reject_unknown_token_id(&tokenizer, &[u32::MAX], || None),
            Ok(())
        );
    }
}
