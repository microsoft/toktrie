use serde::{Deserialize, Serialize};
use std::rc::Rc;

use crate::svob::SimpleVob;
use crate::toktree::{SpecialToken, TokTrie};

pub mod bytes;
pub mod host;
pub mod recognizer;
pub mod rng;
pub mod svob;
pub mod toktree;

pub type TokenId = bytes::TokenId;

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageOp {
    Set,
    Append,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageCmd {
    /// Read variable. Returns StorageResp::ReadVar or StorageResp::VariableMissing.
    ReadVar { name: String },

    /// Write variable.
    /// If `when_version_is == None`, always writes the variable and returns StorageResp::WriteVar.
    /// Otherwise, if the variable has the specified version, it writes the variable
    /// and returns StorageResp::WriteVar.
    /// Otherwise (version conflict), returns either StorageResp::ReadVar or StorageResp::VariableMissing
    /// just like ReadVar would.
    WriteVar {
        name: String,
        value: Vec<u8>,
        op: StorageOp,
        when_version_is: Option<u64>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum StorageResp {
    /// Upon handling the request the variable had the specified value and version number.
    ReadVar { version: u64, value: Vec<u8> },
    /// Upon handling the request the variable was unset.
    VariableMissing {},
    /// The variable has been written, and the new version is returned.
    WriteVar { version: u64 },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitPromptArg {
    pub prompt: Vec<TokenId>,
}

#[repr(transparent)]
#[derive(Serialize, Deserialize, Debug)]
pub struct SeqId(u32);

#[derive(Serialize, Deserialize, Debug)]
pub enum ProcessArg {
    /// Generally, issued after each token generated by the model.
    /// `tokens` is typically just this one token, except for the first call, when
    /// `tokens` is empty, and the cases when fast-forward tokens are used.
    Append { tokens: Vec<TokenId> },

    /// Issued after ProcessResult::Fork.
    /// Use host::self_seq_id() to get the ID of the current sequence.
    Fork { group: Vec<SeqId> },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ProcessResult {
    /// Stop the current sequence.
    /// Similar to strong bias to EOS.
    Stop,

    /// Sample next token in the current sequence, using bias set with `return_logit_bias()`
    SampleWithBias,

    /// First pop `backtrack` tokens,
    /// then force next tokens to be generated to be `ff_tokens`.
    /// `backtrack` can be 0, and `ff_tokens` can be empty but not both.
    Splice {
        backtrack: u32,
        ff_tokens: Vec<TokenId>,
    },

    /// Fork the current sequence into `num_children` sequences (including current one).
    /// `resume_fork(0)` will be called on this VM, while children will be resumed
    /// with `resume_fork(1)` ... `resume_fork(num_children - 1)`
    /// (thus, `Fork {1}` will not create any new sequences).
    Fork { num_children: u32 },

    /// Wait until all listed variables are available for reading,
    /// and all listed sequences have finished executing.
    WaitAll {
        variables: Vec<String>,
        finished: Vec<SeqId>,
    },
}

pub trait AiciVm {
    /// Called with the initial prompt. Has long time limit.
    /// By default ignore prompt.
    fn init_prompt(&mut self, _arg: InitPromptArg) {}

    /// This is the main entry point for the module.
    /// Following calls are issued:
    /// * `Append { tokens: [] }` - to generate bias for the first token of the output
    /// And then any combination of:
    /// * `Append { tokens: [t] }` - when a token `t` is sampled
    /// * `Append { tokens: [t...] }` - after fast-forward
    /// Either way, a bias should be eventually generated.
    fn process(&mut self, arg: ProcessArg) -> ProcessResult;

    fn get_helper(&mut self) -> &mut AiciVmHelper;

    // Internals
    fn aici_process(&mut self) {
        let arg: ProcessArg = serde_json::from_slice(&host::process_arg_bytes()).unwrap();
        let res = self.process(arg);
        let res_bytes = serde_json::to_vec(&res).unwrap();
        host::return_process_result(&res_bytes);
    }

    fn aici_init_prompt(&mut self) {
        let arg: InitPromptArg = serde_json::from_slice(&host::process_arg_bytes()).unwrap();
        self.init_prompt(arg);
    }
}

#[derive(Clone)]
pub struct AiciVmHelper {
    pub allowed_tokens: SimpleVob,
    pub trie: Rc<Box<TokTrie>>,
}

impl AiciVmHelper {
    pub fn new() -> Self {
        let trie = TokTrie::from_host();
        let mut allowed_tokens = SimpleVob::new();
        allowed_tokens.resize(trie.vocab_size() + 1);
        AiciVmHelper {
            allowed_tokens,
            trie: Rc::new(Box::new(trie)),
        }
    }

    pub fn all_disallowed(&mut self) {
        self.allowed_tokens.set_all(false);
    }

    pub fn allow_one(&mut self, tok: TokenId) {
        self.allowed_tokens.allow_token(tok);
    }

    pub fn allow_eos(&mut self) {
        self.allow_one(self.trie.special_token(SpecialToken::EndOfSentence));
    }

    pub fn return_logit_bias(&mut self) -> ProcessResult {
        host::return_logit_bias(&self.allowed_tokens);
        ProcessResult::SampleWithBias
    }
}

/// Expose method as extern "C", usage:
///     expose!(Foo::set_count(n: i32) -> i32);
/// Generates "C" function:
///     set_count(Foo *, i32) -> i32
#[macro_export]
macro_rules! expose {
    ($struct_name:ident :: $method_name:ident ( $($arg:ident : $typ:ty),* ) -> $ret:ty) => {
        #[no_mangle]
        pub extern "C" fn $method_name(self_: *mut $struct_name, $($arg : $typ),*) -> $ret {
            unsafe {
                (&mut *self_).$method_name($($arg),*)
            }
        }
    };
    ($struct_name:ident :: $field:ident :: $method_name:ident ( $($arg:ident : $typ:ty),* ) -> $ret:ty) => {
        #[no_mangle]
        pub extern "C" fn $method_name(self_: *mut $struct_name, $($arg : $typ),*) -> $ret {
            unsafe {
                (&mut *self_).$field.$method_name($($arg),*)
            }
        }
    };
}

#[macro_export]
macro_rules! aici_expose_all {
    ($struct_name:ident, $new:expr) => {
        $crate::expose!($struct_name::aici_process() -> ());
        $crate::expose!($struct_name::aici_init_prompt() -> ());

        #[no_mangle]
        pub extern "C" fn aici_create() -> *mut $struct_name {
            let b = Box::new($new);
            Box::into_raw(b)
        }
    }
}

#[macro_export]
macro_rules! include_bytes_aligned {
    ($align_ty:ty, $path:literal) => {{
        #[repr(C)] // guarantee 'bytes' comes after '_align'
        pub struct AlignedAs<Align, Bytes: ?Sized> {
            pub _align: [Align; 0],
            pub bytes: Bytes,
        }

        // this assignment is made possible by CoerceUnsized
        static ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };

        &ALIGNED.bytes
    }};
}

#[macro_export]
macro_rules! wprintln {
    () => {
        $crate::host::_print("\n")
    };
    ($($arg:tt)*) => {{
        $crate::host::_print(&format!($($arg)*));
        $crate::host::_print("\n");
    }};
}

#[macro_export]
macro_rules! wprint {
    ($($arg:tt)*) => {{
        $crate::host::_print(&format!($($arg)*));
    }};
}
