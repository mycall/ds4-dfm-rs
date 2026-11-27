// The process-wide experiment uses zero-based, inclusive GGUF block numbers.
#[repr(C)]
pub struct LoopRange {
    first: i32,
    count: i32,
}

fn parse_range(text: &str, layers: i32) -> Option<LoopRange> {
    let (first, last) = text.split_once('-')?;
    let parse = |s: &str| {
        s.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| s.parse::<i32>().ok())
            .flatten()
    };
    let (first, last) = (parse(first)?, parse(last)?);
    (first <= last && last < layers).then(|| LoopRange {
        first,
        count: last - first + 1,
    })
}

#[no_mangle]
pub extern "C" fn n1x_loop_range(layers: i32) -> LoopRange {
    match std::env::var("DS4_N1X_LOOP_LAYERS") {
        Err(std::env::VarError::NotPresent) => LoopRange { first: 0, count: 0 },
        value => value
            .ok()
            .and_then(|v| parse_range(&v, layers))
            .unwrap_or(LoopRange {
                first: 0,
                count: -1,
            }),
    }
}

fn extra_layers(repeats: Option<&str>, width: i32, room: i32) -> Option<i32> {
    let count = match repeats {
        None => i32::from(width > 0),
        Some(value) if !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit()) => {
            value.parse().ok()?
        }
        Some(_) => return None,
    };
    if width < 0 || (width == 0 && count != 0) {
        return None;
    }
    width.checked_mul(count).filter(|&total| total <= room)
}

#[no_mangle]
pub extern "C" fn n1x_loop_extra_layers(width: i32, room: i32) -> i32 {
    let value = match std::env::var("DS4_N1X_LOOP_REPEATS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return -1,
    };
    extra_layers(value.as_deref(), width, room).unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inclusive_physical_layers() {
        for (value, want) in [("0-0", (0, 1)), ("15-19", (15, 5)), ("39-39", (39, 1))] {
            let range = parse_range(value, 40).unwrap();
            assert_eq!((range.first, range.count), want);
        }
        for value in ["", "-1-2", "1", "+1-3", "8-7", "1-40", "14-16-18", "8-9 "] {
            assert!(parse_range(value, 40).is_none());
        }
    }

    #[test]
    fn repeat_depth_fits_the_logical_stack() {
        assert_eq!(extra_layers(None, 0, 8), Some(0));
        assert_eq!(extra_layers(None, 4, 8), Some(4));
        assert_eq!(extra_layers(Some("0"), 4, 0), Some(0));
        assert_eq!(extra_layers(Some("2"), 4, 8), Some(8));
        assert_eq!(extra_layers(Some("1"), 0, 8), None);
        for count in ["3", "-1", "+1", "1 ", "", "2147483647", "2147483648"] {
            assert_eq!(extra_layers(Some(count), 4, 8), None);
        }
    }
}
