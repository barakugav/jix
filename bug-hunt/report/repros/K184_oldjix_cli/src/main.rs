// Scratch CLI built against jix 0.0.7 (2b229d2) sources, for K184.
// usage: oldjix-cli write <path> | read <path>
use std::path::Path;

use jix::Array;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(&args[2]);
    match args[1].as_str() {
        "write" => {
            let a = Array::compact_ndarray(&ndarray::array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
            a.write_to_file(path).unwrap();
            println!("jix 0.0.7 wrote {}", path.display());
        }
        "read" => match Array::read_from_file(path) {
            Ok(a) => {
                let shape = a.shape().to_vec();
                let v = a.into_typed::<i32>().unwrap().to_ndarray::<ndarray::IxDyn>().unwrap();
                println!("jix 0.0.7 read OK: shape {shape:?} values {:?}", v.iter().collect::<Vec<_>>());
            }
            Err(e) => {
                println!("jix 0.0.7 read FAILED: kind {:?}: {e}", e.kind());
                std::process::exit(1);
            }
        },
        _ => panic!("bad mode"),
    }
}
