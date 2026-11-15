use std::io::{Seek, SeekFrom};

// copied from std::io::Seek::stream_len implementation
pub(crate) fn stream_len<T: Seek + ?Sized>(self_: &mut T) -> std::io::Result<u64> {
    let old_pos = self_.stream_position()?;
    let len = self_.seek(SeekFrom::End(0))?;

    // Avoid seeking a third time when we were already at the end of the
    // stream. The branch is usually way cheaper than a seek operation.
    if old_pos != len {
        self_.seek(SeekFrom::Start(old_pos))?;
    }

    Ok(len)
}
