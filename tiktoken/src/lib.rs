use std::{collections::HashMap, sync::Arc};

use anyhow::{bail, Result};
use tiktoken_rs::{cl100k_base, o200k_base, CoreBPE};
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

fn get_tokenizer(name: &str) -> Result<(CoreBPE, usize)> {
    match name {
        "o200k_base" => Ok((o200k_base()?, 199998)),
        "cl100k_base" => Ok((cl100k_base()?, 100256)),
        // TODO add llama3 tokenizer
        _ => bail!(
            "Unknown tiktoken tokenizer: {}; allowed options o200k_base and cl100k_base",
            name
        ),
    }
}

pub struct TikTokenEnv {
    tokenizer: CoreBPE,
    tok_trie: TokTrie,
    special_tokens: HashMap<String, TokenId>,
}

pub struct TikTokenConfig {
    pub name: String,
    pub eos_token: TokenId,
    pub vocab_size_override: Option<usize>,
    pub special_tokens: HashMap<String, TokenId>,
}

impl TikTokenEnv {
    pub fn new(config: TikTokenConfig) -> Result<Self> {
        let (tokenizer, mut n_vocab) = get_tokenizer(&config.name)?;

        let mut tokens = Vec::with_capacity(n_vocab);
        for i in 0..n_vocab {
            let buf = tokenizer._decode_native(&[i]);
            tokens.push(buf);
        }

        let max_spec = *config.special_tokens.values().max().unwrap_or(&0) as usize;
        if max_spec >= n_vocab {
            n_vocab = max_spec + 1;
        }

        if let Some(vocab_size_override) = config.vocab_size_override {
            if vocab_size_override < n_vocab {
                bail!(
                    "vocab_size_override {} too low (need at least {})",
                    vocab_size_override,
                    n_vocab
                );
            }
            n_vocab = vocab_size_override;
        }

        tokens.resize(n_vocab, vec![]);

        for (name, id) in &config.special_tokens {
            let mut name_vec = name.as_bytes().to_vec();
            name_vec.insert(0, TokTrie::SPECIAL_TOKEN_PREFIX_BYTE);
            tokens[*id as usize] = name_vec;
        }

        let tokrxinfo = TokRxInfo::new(n_vocab as u32, config.eos_token);
        let tok_trie = TokTrie::from(&tokrxinfo, &tokens);

        Ok(Self {
            tokenizer,
            tok_trie,
            special_tokens: config.special_tokens,
        })
    }

    pub fn get_tokenizer(&self) -> &CoreBPE {
        &self.tokenizer
    }

    pub fn to_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for TikTokenEnv {
    fn stop(&self) -> ! {
        panic!("stop() called");
    }

    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_special(&self, s: &str) -> Vec<TokenId> {
        if let Some(&id) = self.special_tokens.get(s) {
            vec![id]
        } else {
            self.tokenize_bytes(s.as_bytes())
        }
    }

    fn tokenize_bytes_prefix(&self, s: &[u8]) -> Vec<TokenId> {
        let mut idx = 0;
        let ff = TokTrie::SPECIAL_TOKEN_PREFIX_BYTE;
        let mut result = Vec::new();
        while idx < s.len() {
            let normal_len = s[idx..]
                .iter()
                .position(|&x| x == ff)
                .unwrap_or(s.len() - idx);
            if normal_len != 0 {
                result.extend_from_slice(&self.tokenize_bytes(&s[idx..idx + normal_len]));
                idx += normal_len;
            }
            idx += 1; // skip ff
            if idx + 3 < s.len() && s[idx] == '<' as u8 {
                let spec_len = s[idx..std::cmp::min(s.len(), idx + 100)]
                    .iter()
                    .position(|&x| x == '>' as u8);
                if let Some(mut spec_len) = spec_len {
                    spec_len += 1;
                    let spec_token = String::from_utf8_lossy(&s[idx..idx + spec_len]);
                    if let Some(&id) = self.special_tokens.get(spec_token.as_ref()) {
                        result.push(id);
                        idx += spec_len;
                    }
                }
            }
        }

        result
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tokenizer
                .encode_ordinary(s)
                .into_iter()
                .map(|x| x as TokenId)
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_tiktoken_env_initialization() {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<eos>".to_string(), 100001); // Example special token ID

        let config = TikTokenConfig {
            name: "cl100k_base".to_string(),
            eos_token: 100256,
            vocab_size_override: Some(100300),
            special_tokens: special_tokens.clone(),
        };

        let env = TikTokenEnv::new(config).expect("Failed to initialize TikTokenEnv");

        // Check if special tokens were set correctly
        for (token, id) in &special_tokens {
            assert_eq!(env.special_tokens.get(token), Some(id));
        }
    }
}
