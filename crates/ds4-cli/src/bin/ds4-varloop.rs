//! Standalone native Varloop text runner. The token-ID path also works on fixture GGUFs.
use ds4_core::{varloop::Session, GgufFile, ModelFamily, Vocab};
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

const HELP: &str = concat!("ds4-varloop -m MODEL.gguf (--tokens 1,2,3 | -p TEXT) [--passes 1..5] ",
    "[-n COUNT] [--backend cpu|cuda] [--dump-logits FILE] [--timings]\n",
    "-p takes raw completion text. --dump-logits writes little-endian FP32 logits after each input token.");

fn run() -> Result<(), String> {
    let mut it = env::args().skip(1);
    let (mut path, mut prompt, mut ids, mut dump) = (None, None, None, None);
    let mut passes = None;
    let mut count = 1usize;
    let mut cuda = cfg!(feature = "varloop-cuda");
    let mut timings = false;
    while let Some(arg) = it.next() {
        if arg == "--help" || arg == "-h" {
            println!("{HELP}");
            return Ok(());
        }
        if arg == "--timings" {
            timings = true;
            continue;
        }
        let v = it
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "-m" | "--model" => path = Some(PathBuf::from(v)),
            "-p" => prompt = Some(v),
            "--tokens" => {
                ids = Some(
                    v.split(',')
                        .map(|t| t.parse::<u32>())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| "invalid token ID list")?,
                )
            }
            "--passes" => passes = Some(v.parse().map_err(|_| "invalid pass count")?),
            "-n" => count = v.parse().map_err(|_| "invalid generation length")?,
            "--backend" => {
                cuda = match v.as_str() {
                    "cpu" => false,
                    "cuda" => true,
                    _ => return Err("expected cpu or cuda".into()),
                }
            }
            "--dump-logits" => dump = Some(PathBuf::from(v)),
            _ => return Err(format!("unknown option {arg}")),
        }
    }
    let path = path.ok_or(HELP)?;
    if ids.is_some() == prompt.is_some() {
        return Err("choose --tokens or -p".into());
    }
    let vocab = if prompt.is_some() {
        let file = GgufFile::open(&path).map_err(|e| e.to_string())?;
        Some(Vocab::load(&file, ModelFamily::Inkling2Vq).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let tokens = match (ids, &vocab, prompt) {
        (Some(ids), _, _) => ids,
        (_, Some(v), Some(p)) => v.encode_text(&p).into_iter().map(|t| t as u32).collect(),
        _ => unreachable!(),
    };
    if tokens.is_empty() {
        return Err("empty token sequence".into());
    }
    let mut model = Session::open(&path, passes, cuda)?;
    // Refuse to truncate the mapped model or a previously saved run.
    let mut writer = dump
        .map(|p| File::create_new(p).map(BufWriter::new))
        .transpose()
        .map_err(|e| e.to_string())?;
    let mut logits = None;
    let start = Instant::now();
    for (i, &t) in tokens.iter().enumerate() {
        logits = model.step(t, writer.is_some() || i + 1 == tokens.len())?;
        if let Some(out) = &mut writer {
            for z in logits.as_ref().unwrap() {
                out.write_all(&z.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
    }
    if let Some(mut out) = writer {
        out.flush().map_err(|e| e.to_string())?;
    }
    let mut out = std::io::stdout().lock();
    let decode_start = Instant::now();
    let mut decode_steps = 0;
    for i in 0..count {
        let values = logits.take().unwrap();
        if !values.iter().all(|v| v.is_finite()) {
            return Err("non-finite logits".into());
        }
        let best = values
            .iter()
            .enumerate()
            .reduce(|a, b| if b.1 > a.1 { b } else { a })
            .unwrap()
            .0;
        if let Some(v) = &vocab {
            if v.is_stop(best as i32) {
                break;
            }
            out.write_all(&v.token_text(best as i32))
                .map_err(|e| e.to_string())?;
        } else {
            write!(out, "{}{best}", if i == 0 { "" } else { "," }).map_err(|e| e.to_string())?;
        }
        if i + 1 < count {
            logits = model.step(best as u32, true)?;
            decode_steps += 1;
        }
    }
    if timings {
        eprintln!(
            "prefill {} tokens {:.3} ms, decode {} steps {:.3} ms",
            tokens.len(),
            decode_start.duration_since(start).as_secs_f64() * 1000.,
            decode_steps,
            decode_start.elapsed().as_secs_f64() * 1000.
        );
    }
    writeln!(out).map_err(|e| e.to_string())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
