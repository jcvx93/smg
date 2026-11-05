//! The bound on the token ids a request names: the model's vocabulary, its
//! added tokens and the embedding table the model's workers report. A
//! pretokenized prompt's ids and a `logit_bias` map's keys are judged by the
//! one rule here; each caller words its own refusal.

use llm_tokenizer::traits::Tokenizer;

use crate::worker::WorkerRegistry;

/// The worker label that carries the engine's embedding-table size: `vocab_size` of the
/// engine's model info, kept on every worker that reports it.
const VOCAB_SIZE_LABEL: &str = "vocab_size";

/// The embedding-table size the model's workers report, the smallest among those that report
/// one (an id valid on a larger table is not on a smaller); `None` when no worker reports it.
/// Read only after an id the tokenizer does not know, so a valid request never pays for it.
pub(crate) fn reported_embedding_rows(
    worker_registry: &WorkerRegistry,
    model_id: &str,
) -> Option<usize> {
    worker_registry
        .get_by_model(model_id)
        .iter()
        .filter_map(|worker| {
            worker
                .metadata()
                .spec
                .labels
                .get(VOCAB_SIZE_LABEL)
                .and_then(|rows| rows.parse::<usize>().ok())
        })
        .min()
}

/// An id a request named that is not a token of the model: its position among
/// the ids given, its value, and the bound it missed, worded for a refusal.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct UnknownTokenId {
    pub(crate) index: usize,
    pub(crate) id: u32,
    pub(crate) bound: String,
}

/// The first id that can reach no row of the engine's embedding table, with the bound it
/// missed; `None` when every id is a token of the model.
///
/// An id below the tokenizer's base vocabulary is known by its range alone; one above it is an
/// added token (the special tokens live there) and is resolved by lookup, so the common id costs
/// one comparison and only the rare one a map read. The engines index their embeddings by these
/// ids and the headless lanes run none of the checks an engine's own HTTP front end would, so an
/// id past the table must not leave the gateway: on the engine it is an out-of-range index, not a
/// refused request. The table is padded past the tokenizer's last id on most models and an
/// engine samples over the padded logits, so an id the tokenizer does not know is judged by the
/// table size the model's workers report (`embedding_rows`, read once, only then): below it the
/// id passes, as it would on the engine. When no worker reports a table the tokenizer's ids are
/// the bound, the one case where the gateway is stricter than the engine; a tokenizer that
/// reports no vocabulary either skips the bound rather than refusing every id.
pub(crate) fn first_unknown_token_id(
    tokenizer: &dyn Tokenizer,
    ids: &[u32],
    embedding_rows: impl FnOnce() -> Option<usize>,
) -> Option<UnknownTokenId> {
    let base_vocab = tokenizer.vocab_size();
    let known = |id: u32| {
        usize::try_from(id).is_ok_and(|id| id < base_vocab) || tokenizer.id_to_token(id).is_some()
    };
    let first_unknown = if base_vocab == 0 {
        // Without a vocabulary every id would be a lookup: the table alone judges them.
        0
    } else {
        ids.iter().position(|&id| !known(id))?
    };
    match embedding_rows() {
        Some(rows) => {
            let inside = |id: u32| usize::try_from(id).is_ok_and(|id| id < rows);
            let offset = ids[first_unknown..]
                .iter()
                .position(|&id| !inside(id) && !known(id))?;
            Some(UnknownTokenId {
                index: first_unknown + offset,
                id: ids[first_unknown + offset],
                bound: format!("its embedding table holds {rows} rows"),
            })
        }
        None if base_vocab == 0 => None,
        None => Some(UnknownTokenId {
            index: first_unknown,
            id: ids[first_unknown],
            bound: format!("its vocabulary holds {base_vocab} tokens plus its added tokens"),
        }),
    }
}
