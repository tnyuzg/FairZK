use std::vec;

use algebra::{
    //derive::{Field, Prime},
    BabyBear,
    DenseMultilinearExtension,
    Field,
    FieldUniformSampler,
    ListOfProductsOfPolynomials,
    MultilinearExtension,
};
use num_traits::Zero;
use rand::thread_rng;
use rand_distr::Distribution;
use std::rc::Rc;

macro_rules! field_vec {
    ($t:ty; $elem:expr; $n:expr)=>{
        vec![<$t>::new($elem);$n]
    };
    ($t:ty; $($x:expr),+ $(,)?) => {
        vec![$(<$t>::new($x)),+]
    }
}

// field type
type FF = BabyBear;
type PolyFf = DenseMultilinearExtension<FF>;

fn evaluate_mle_data_array<F: Field>(data: &[F], point: &[F]) -> F {
    if data.len() != (1 << point.len()) {
        panic!("Data size mismatch with number of variables.")
    }
    let nv = point.len();
    let mut a = data.to_vec();

    for i in 1..nv + 1 {
        let r = point[i - 1];
        for b in 0..(1 << (nv - i)) {
            a[b] = a[b << 1] * (F::one() - r) + a[(b << 1) + 1] * r;
        }
    }

    a[0]
}

// P(0,0,0) P(1,0,0) P(0,1,0) P(1,1,0) P(0,0,1) P(1,0,1) P(0,1,1) P(1,1,1)
// 1        2        3        4        5        6        7        8
#[test]
fn test_partial_evaluate() {
    let poly = PolyFf::from_vec(3, field_vec! {FF; 1, 2, 3, 4, 5, 6, 7, 8});
    let point = field_vec! {FF; 0,1};
    let partial = poly.fix_variables_back(&point);
    //let partial = poly.fix_variables_front(&point);
    dbg!(&partial);
    //assert_eq!(partial.evaluate(&point[1..]), FF::new(3));
}

#[test]
fn evaluate_mle_at_a_point() {
    let poly = PolyFf::from_vec(2, field_vec! {FF; 1, 2, 3, 4});

    let point = vec![FF::new(0), FF::new(1)];
    assert_eq!(poly.evaluate(&point), FF::new(3));
}

#[test]
fn evaluate_mle_at_a_random_point() {
    let mut rng = thread_rng();
    let poly = PolyFf::random(2, &mut rng);
    let uniform = <FieldUniformSampler<FF>>::new();
    let point: Vec<_> = (0..2).map(|_| uniform.sample(&mut rng)).collect();
    assert_eq!(
        poly.evaluate(&point),
        evaluate_mle_data_array(&poly.evaluations, &point),
    );
}

#[test]
fn mle_arithmetic() {
    const NV: usize = 10;
    let mut rng = thread_rng();
    let uniform = <FieldUniformSampler<FF>>::new();
    for _ in 0..20 {
        let point: Vec<_> = (0..NV).map(|_| uniform.sample(&mut rng)).collect();
        let poly1 = PolyFf::random(NV, &mut rng);
        let poly2 = PolyFf::random(NV, &mut rng);
        let v1 = poly1.evaluate(&point);
        let v2 = poly2.evaluate(&point);
        // test add
        assert_eq!((&poly1 + &poly2).evaluate(&point), v1 + v2);
        // test sub
        assert_eq!((&poly1 - &poly2).evaluate(&point), v1 - v2);
        // test negate
        assert_eq!(-poly1.evaluate(&point), -v1);
        // test add assign
        {
            let mut poly1 = poly1.clone();
            poly1 += &poly2;
            assert_eq!(poly1.evaluate(&point), v1 + v2);
        }
        // test sub assign
        {
            let mut poly1 = poly1.clone();
            poly1 -= &poly2;
            assert_eq!(poly1.evaluate(&point), v1 - v2);
        }
        // test add assign with scalar
        {
            let mut poly1 = poly1.clone();
            let scalar = uniform.sample(&mut rng);
            poly1 += (scalar, &poly2);
            assert_eq!(poly1.evaluate(&point), v1 + scalar * v2);
        }
        // test additive identity
        {
            assert_eq!(&poly1 + &PolyFf::zero(), poly1);
            assert_eq!((&PolyFf::zero() + &poly1), poly1);
        }
    }
}

#[test]
fn evaluate_lists_of_products_at_a_point() {
    let nv = 2;
    let mut poly = ListOfProductsOfPolynomials::new(nv);
    let products = vec![field_vec!(FF; 1, 2, 3, 4), field_vec!(FF; 5, 4, 2, 9)];
    let products: Vec<Rc<DenseMultilinearExtension<FF>>> = products
        .into_iter()
        .map(|x| Rc::new(DenseMultilinearExtension::from_vec(nv, x)))
        .collect();
    let coeff = FF::new(4);
    poly.add_product(products, coeff);

    let point = field_vec!(FF; 0, 1);
    assert_eq!(poly.evaluate(&point), FF::new(24));
}

#[test]
fn evaluate_lists_of_products_with_op_at_a_point() {
    let nv = 2;
    let mut poly = ListOfProductsOfPolynomials::new(nv);
    let products = vec![field_vec!(FF; 1, 2, 3, 4), field_vec!(FF; 1, 2, 3, 4)];
    let products: Vec<Rc<DenseMultilinearExtension<FF>>> = products
        .into_iter()
        .map(|x| Rc::new(DenseMultilinearExtension::from_vec(nv, x)))
        .collect();
    let coeff = FF::new(4);
    let op_coefficient = vec![(FF::new(2), FF::new(0)), (FF::new(1), FF::new(3))];
    // coeff \cdot [2f \cdot (g + 3)]
    poly.add_product_with_linear_op(products, &op_coefficient, coeff);
    // 4 * [2*2 * (2+3)] = 80
    let point = field_vec!(FF; 1, 0);
    assert_eq!(poly.evaluate(&point), FF::new(80));
}
