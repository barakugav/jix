use jix::Array;
use ndarray::{array, Ix2};

#[test]
fn smoke() {
    let a = Array::compact_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    let s = a.view().exp().sum(0);
    let _ = s.to_ndarray::<ndarray::Ix1>().unwrap();
    let p = a.permute_dims(&[1, 0]);
    assert_eq!(p.to_ndarray::<Ix2>().unwrap(), array![[1.0f32, 4.0], [2.0, 5.0], [3.0, 6.0]]);
}
