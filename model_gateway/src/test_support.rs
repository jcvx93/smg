//! What the crate's unit tests and its integration binaries share: fixtures
//! with no home in one subsystem. Compiled for the crate's own test targets
//! only (`cfg(test)`, or the `test-util` feature the self dev-dependency
//! turns on for them), never into a production build.

use std::{
    collections::BTreeMap,
    net::{Ipv4Addr, SocketAddr},
    sync::atomic::{AtomicUsize, Ordering},
};

use k8s_openapi::{
    api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
    apimachinery::pkg::apis::meta::v1::ObjectMeta,
};
use llm_tokenizer::{
    traits::{Encoding, Tokenizer},
    SpecialTokens,
};
use tokio::net::TcpSocket;

/// A loopback address nothing answers at, refused for as long as this value
/// lives.
///
/// The socket behind it is bound and never listening: the host's own stack
/// refuses every connection at once, the way a dead worker's port does, and
/// the port stays taken, so no other test's `bind("127.0.0.1:0")` can draw
/// it in the meantime. A listener bound and dropped for the same purpose
/// gives its port back, and a neighbouring test's stub then answers in the
/// unreachable side's place: a `404` or `405` where a transport failure was
/// expected, or a request held until the test's bound.
///
/// Keep the value in a named binding for the whole case: a `_` pattern or a
/// temporary drops it, and the port is free again.
#[must_use = "the address is refused only while this value lives"]
pub struct RefusedAddress {
    _reservation: TcpSocket,
    addr: SocketAddr,
}

impl RefusedAddress {
    /// Reserve a loopback port of the system's choice, and refuse it.
    #[expect(
        clippy::expect_used,
        reason = "a test fixture: a loopback socket the system cannot give is a test failure"
    )]
    pub fn reserve() -> Self {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let addr = socket.local_addr().expect("the bound address");
        Self {
            _reservation: socket,
            addr,
        }
    }

    /// The address, `127.0.0.1:<port>`.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The port alone, for a worker registered with a bootstrap port.
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    /// The address as a plain HTTP URL, `http://127.0.0.1:<port>`.
    pub fn http_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

/// A running, ready pod named `name` at `ip`, carrying `labels`: the shape
/// the Kubernetes watchers' tests feed to their label selectors.
pub fn make_labeled_pod(name: &str, ip: &str, labels: &[(&str, &str)]) -> Pod {
    let mut label_map = BTreeMap::new();
    for &(k, v) in labels {
        label_map.insert(k.to_string(), v.to_string());
    }
    Pod {
        metadata: ObjectMeta {
            name: Some(name.to_string()),
            uid: Some(format!("uid-{name}")),
            labels: Some(label_map),
            ..Default::default()
        },
        spec: Some(PodSpec::default()),
        status: Some(PodStatus {
            pod_ip: Some(ip.to_string()),
            phase: Some("Running".to_string()),
            conditions: Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "True".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]),
            ..Default::default()
        }),
    }
}

/// A tokenizer whose tokens are characters (`id = char as u32`): the
/// streaming tests' and the stream benchmark's, so an engine stream scripted
/// as text decodes with no model's vocabulary behind it.
#[derive(Default)]
pub struct CharacterTokenizer {
    special_tokens: SpecialTokens,
    /// How many times the eager decode step ran (a test reads it).
    pub(crate) eager_steps: AtomicUsize,
    /// The ids `eos_token_ids` reports (none by default).
    eos: Vec<u32>,
}

impl CharacterTokenizer {
    /// A tokenizer whose EOS ids are `eos`: the gateway's stop decoder takes
    /// them as stops, as it does a model's.
    pub fn with_eos(eos: Vec<u32>) -> Self {
        Self {
            eos,
            ..Self::default()
        }
    }
}

impl llm_tokenizer::traits::Encoder for CharacterTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(text.chars().map(u32::from).collect()))
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }
}

impl llm_tokenizer::traits::Decoder for CharacterTokenizer {
    fn decode_step_eager(
        &self,
        token_id: u32,
        ids: &mut Vec<u32>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> anyhow::Result<Option<String>> {
        self.eager_steps.fetch_add(1, Ordering::Relaxed);
        self.decode_step(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        ids.iter()
            .map(|id| char::from_u32(*id).ok_or_else(|| anyhow::anyhow!("invalid character token")))
            .collect()
    }
}

impl Tokenizer for CharacterTokenizer {
    fn vocab_size(&self) -> usize {
        128
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.chars().next().map(u32::from)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        char::from_u32(id).map(|character| character.to_string())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use super::RefusedAddress;

    /// A connection to the address is refused at once, and no other bind
    /// takes the port while the value lives.
    #[tokio::test]
    async fn the_address_refuses_connections_and_keeps_its_port() {
        let refused = RefusedAddress::reserve();
        let error = tokio::net::TcpStream::connect(refused.addr())
            .await
            .expect_err("nothing answers there");
        assert_eq!(error.kind(), ErrorKind::ConnectionRefused, "{error}");
        let taken = std::net::TcpListener::bind(refused.addr()).expect_err("the port is held");
        assert_eq!(taken.kind(), ErrorKind::AddrInUse, "{taken}");
        assert_eq!(
            refused.http_url(),
            format!("http://127.0.0.1:{}", refused.port())
        );
    }
}
