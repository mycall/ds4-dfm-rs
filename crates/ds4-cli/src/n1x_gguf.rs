//! One-shot inference for the split DeepSeek 4.1 GGUF on Windows ARM64.
use super::{is_rendered_chat_prompt, spec_sampler, ShadowArgs, TokenPrinter};
use ds4_core::n1x_gguf::Session;
use ds4_core::{chat_template::Template, Backend, GgufFile, ModelFamily, Vocab};
use serde_json::json;
use std::io::{self, Write};
use std::path::Path;

pub(super) fn matches(path: &Path) -> bool {
    path.is_file()
        && GgufFile::open(path)
            .is_ok_and(|file| file.get_string("general.architecture") == Some(b"deepseek41"))
}

pub(super) fn generate(args: &ShadowArgs, path: &str, prompt: &str) -> Result<i32, String> {
    if args.mtp.is_some()
        || args.mtp_draft != 1
        || args.dspark.is_some()
        || args.dist.role != ds4_dist::Role::None
        || args.serving_check
        || args.gen_ids.is_some()
        || args.engram_dir.is_some()
        || args.no_engram
        || args.vision.is_some()
        || !args.images.is_empty()
        || args.vision_golden.is_some()
        || args.zchain.is_some()
        || args.posttrain.is_some()
        || args.mem_budget_mb.is_some()
        || args.no_graph
        || args.emit_trace
        || args.v41_prof
        || args.lifecycle_only
        || !args.ssd_options.is_empty()
        || args.dry_multiplier != 0.0
        || args.dump_logprobs.is_some()
        || args.dump_logits.is_some()
        || !args.tokens.is_empty()
        || args.predict > 0
    {
        return Err("the N1X deepseek41 GGUF backend supports one-shot text generation".into());
    }
    let gpu = match args.backend {
        Backend::Cpu => false,
        Backend::Cuda => true,
        _ => return Err("N1X deepseek41 requires CUDA or CPU".into()),
    };
    let model = Path::new(path);
    let family = ModelFamily::DeepSeek4;
    let vocab = Vocab::load_path(model, family).map_err(|e| e.to_string())?;
    let rendered = if is_rendered_chat_prompt(prompt) {
        prompt.to_owned()
    } else {
        let template = Template::from_model(model)
            .map_err(|e| e.to_string())?
            .ok_or("missing DeepSeek 4.1 chat template")?;
        let mut messages = Vec::new();
        if !args.system.is_empty() {
            messages.push(json!({"role": "system", "content": args.system}));
        }
        messages.push(json!({"role": "user", "content": prompt}));
        template
            .render(&json!({
                "messages": messages,
                "enable_thinking": !args.nothink,
                "add_generation_prompt": true
            }))
            .map_err(|e| e.to_string())?
    };
    let mut tokens = vocab.encode_rendered_chat(&rendered);
    if tokens.is_empty() || tokens.len() >= args.ctx as usize {
        return Err(format!(
            "prompt length {} exceeds available context {}",
            tokens.len(),
            args.ctx
        ));
    }
    let limit = args.n_predict.min(args.ctx - tokens.len() as i32);
    let mut session = Session::open(path, gpu, args.ctx, args.threads, vocab.n_vocab())?;
    let mut out = io::stdout().lock();
    let mut printer = TokenPrinter::new(family, false);
    let sampler = spec_sampler(args);
    let mut rng = args.seed;
    for _ in 0..limit {
        let id = session.sample(&tokens, &sampler, &mut rng)?;
        if vocab.is_stop(id) {
            break;
        }
        printer
            .write_token(&mut out, id, &vocab.token_text(id))
            .and_then(|_| out.flush())
            .map_err(|e| e.to_string())?;
        tokens.clear();
        tokens.push(id);
    }
    printer.finish(&mut out).map_err(|e| e.to_string())?;
    Ok(0)
}
