//! This module defines some useful utils that may invoked by piop.
use std::{collections::HashMap, rc::Rc};
use std::{mem, vec};

use algebra::{AsFrom, PrimeField};
use algebra::{DenseMultilinearExtension, Field};
use nalgebra::{DMatrix, DVector};
use rand::Rng;

/// Generate MLE of the ideneity function eq(u,x) for x \in \{0, 1\}^dim
#[inline]
pub fn gen_identity_poly<F: Field>(u: &[F]) -> DenseMultilinearExtension<F> {
    let dim = u.len();
    let mut evaluations: Vec<_> = vec![F::zero(); 1 << dim];
    evaluations[0] = F::one();
    for i in 0..dim {
        // The index represents a point in {0,1}^`num_vars` in little endian form.
        // For example, `0b1011` represents `P(1,1,0,1)`
        let u_i_rev = u[dim - i - 1];
        for b in (0..(1 << i)).rev() {
            evaluations[(b << 1) + 1] = evaluations[b] * u_i_rev;
            evaluations[b << 1] = evaluations[b] * (F::one() - u_i_rev);
        }
    }
    DenseMultilinearExtension::from_vec(dim, evaluations)
}

/// Evaluate eq(u, v) = \prod_i (u_i * v_i + (1 - u_i) * (1 - v_i))
#[inline]
pub fn eval_identity_poly<F: Field>(u: &[F], v: &[F]) -> F {
    assert_eq!(u.len(), v.len());
    let mut evaluation = F::one();
    for (u_i, v_i) in u.iter().zip(v) {
        evaluation *= *u_i * *v_i + (F::one() - *u_i) * (F::one() - *v_i);
    }
    evaluation
}

#[inline]
pub fn random_positive_u64_vec(length: usize, range: u64) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..length).map(|_| rng.gen_range(0..range)).collect()
}

#[inline]
pub fn random_absolute_u64_vec<F: Field>(length: usize, range: u64) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let p: u64 = F::MODULUS_VALUE.into();
    (0..length)
        .map(|_| {
            if rng.gen_bool(0.5) {
                rng.gen_range(0..range)
            } else {
                rng.gen_range(p - range + 1u64..p)
            }
        })
        .collect()
}

pub fn f64_to_u64<F: Field>(x: f64, frac_bits: usize, int_bits: usize) -> u64 {
    assert!(x.abs() < 2_f64.powi(int_bits as i32));
    let q = F::MODULUS_VALUE.into();
    let rounded = (x.abs() * 2_f64.powi(frac_bits as i32)).round() as u64;
    if x < 0.0 {
        (q - rounded) as u64
    } else {
        rounded
    }
}

pub fn field_to_f64<F: PrimeField>(vec: &[F], frac_bits: usize, int_bits: usize) -> Vec<f64> {
    let q: u64 = F::MODULUS_VALUE.into();
    let f_shift = 2 ^ frac_bits;
    vec.iter()
        .map(|&x| {
            let x_u64 = x.value().into() as u64;
            let x_f64 = if x_u64 > q / 2 {
                -((q - x_u64) as f64) / f_shift as f64
            } else {
                x_u64 as f64 / f_shift as f64
            };
            assert!(x_f64.abs() < 2_f64.powi(int_bits as i32));
            x_f64
        })
        .collect()
}

pub fn u64_to_field<F>(vec: &[u64]) -> Vec<F>
where
    F: Field,
{
    vec.iter().map(|&x| F::new(F::Value::as_from(x))).collect()
}

pub fn field_to_u64<F>(vec: &[F]) -> Vec<u64>
where
    F: PrimeField,
{
    vec.iter().map(|&x| x.value().into()).collect()
}

pub fn field_to_u128<F>(vec: &[F]) -> Vec<u128>
where
    F: PrimeField,
{
    vec.iter().map(|&x| x.value().into() as u128).collect()
}

pub fn field_to_i128<F>(vec: &[F]) -> Vec<i128>
where
    F: PrimeField,
{
    vec.iter()
        .map(|&x| {
            let x_u128 = x.value().into() as u128;
            let q = F::MODULUS_VALUE.into() as u128;
            if x_u128 > q / 2 {
                -((q - x_u128) as i128)
            } else {
                x_u128 as i128
            }
        })
        .collect()
}

pub fn i128_to_field<F>(vec: &[i128]) -> Vec<F>
where
    F: PrimeField,
{
    vec.iter()
        .map(|&x| {
            if x < 0 {
                F::new(F::Value::as_from(
                    (F::MODULUS_VALUE.into() as i128 + x) as u64,
                ))
            } else {
                F::new(F::Value::as_from(x as u64))
            }
        })
        .collect()
}

pub fn u32_to_field<F>(vec: &[u32]) -> Vec<F>
where
    F: Field,
    F::Value: From<u32> + Into<u32>,
{
    vec.iter().map(|&x| F::new(F::Value::from(x))).collect()
}

#[inline]
pub fn vec_to_poly<F: Field>(vec: &[F]) -> Rc<DenseMultilinearExtension<F>> {
    let num_vars = vec.len().ilog2() as usize;
    debug_assert!(vec.len() == (1 << num_vars));
    Rc::new(DenseMultilinearExtension::from_slice(num_vars, vec))
}

#[inline]
pub fn vec_to_polys<F: Field>(num_vars: usize, vec: &[F]) -> Vec<Rc<DenseMultilinearExtension<F>>> {
    debug_assert!(vec.len() % (1 << num_vars) == 0);
    vec.chunks_exact(1 << num_vars)
        .map(|evals| Rc::new(DenseMultilinearExtension::from_slice(num_vars, evals)))
        .collect()
}

#[inline]
pub fn polys_to_vec<F: Field>(polys: &[Rc<DenseMultilinearExtension<F>>]) -> Vec<F> {
    polys
        .iter()
        .flat_map(|poly| poly.evaluations.iter().cloned())
        .collect()
}

#[inline]
pub fn vec_to_named_polys<F: Field>(
    num_vars: usize,
    vec: &[F],
    name: &[String],
) -> Vec<Rc<DenseMultilinearExtension<F>>> {
    debug_assert!(vec.len() % (1 << num_vars) == 0);
    debug_assert!(vec.len() / (1 << num_vars) == name.len());
    vec.chunks_exact(1 << num_vars)
        .zip(name.iter())
        .map(|(evals, name)| {
            Rc::new(DenseMultilinearExtension::from_named_slice(
                name, num_vars, evals,
            ))
        })
        .collect()
}

#[inline]
pub fn vec_to_named_poly<F: Field>(name: &String, vec: &[F]) -> Rc<DenseMultilinearExtension<F>> {
    let num_vars = vec.len().ilog2() as usize;
    debug_assert!(vec.len() == (1 << num_vars));
    Rc::new(DenseMultilinearExtension::from_named_slice(
        name, num_vars, vec,
    ))
}

#[inline]
pub fn combine_suboracle_evals<F: Field>(evals: &[F], random_point: &[F]) -> F {
    let eq_at_r = gen_identity_poly(random_point);
    let randomized_eval = evals
        .iter()
        .zip(eq_at_r.iter())
        .fold(F::zero(), |acc, (eval, coeff)| acc + *eval * *coeff);
    randomized_eval
}

#[inline]
pub fn binary_vec<F: Field>(n: usize) -> Vec<Vec<F>> {
    let num_bit = n.next_power_of_two().ilog2() as usize;
    (0..n)
        .map(|value| {
            (0..num_bit)
                .map(|i| {
                    if (value >> i) & 1 == 1 {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect() // 将每个位收集到 `Vec<F>`
        })
        .collect() // 将所有 `Vec<F>` 收集到 `Vec<Vec<F>>`
}

// credit@Plonky3
/// Batch multiplicative inverses with Montgomery's trick
/// This is Montgomery's trick. At a high level, we invert the product of the given field
/// elements, then derive the individual inverses from that via multiplication.
///
/// The usual Montgomery trick involves calculating an array of cumulative products,
/// resulting in a long dependency chain. To increase instruction-level parallelism, we
/// compute WIDTH separate cumulative product arrays that only meet at the end.
///
/// # Panics
/// Might panic if asserts or unwraps uncover a bug.
pub fn batch_inverse<F: Field>(x: &[F]) -> Vec<F> {
    // Higher WIDTH increases instruction-level parallelism, but too high a value will cause us
    // to run out of registers.
    const WIDTH: usize = 4;
    // JN note: WIDTH is 4. The code is specialized to this value and will need
    // modification if it is changed. I tried to make it more generic, but Rust's const
    // generics are not yet good enough.

    // Handle special cases. Paradoxically, below is repetitive but concise.
    // The branches should be very predictable.
    let n = x.len();
    if n == 0 {
        return Vec::new();
    } else if n == 1 {
        return vec![x[0].inv()];
    } else if n == 2 {
        let x01 = x[0] * x[1];
        let x01inv = x01.inv();
        return vec![x01inv * x[1], x01inv * x[0]];
    } else if n == 3 {
        let x01 = x[0] * x[1];
        let x012 = x01 * x[2];
        let x012inv = x012.inv();
        let x01inv = x012inv * x[2];
        return vec![x01inv * x[1], x01inv * x[0], x012inv * x01];
    }
    debug_assert!(n >= WIDTH);

    // Buf is reused for a few things to save allocations.
    // Fill buf with cumulative product of x, only taking every 4th value. Concretely, buf will
    // be [
    //   x[0], x[1], x[2], x[3],
    //   x[0] * x[4], x[1] * x[5], x[2] * x[6], x[3] * x[7],
    //   x[0] * x[4] * x[8], x[1] * x[5] * x[9], x[2] * x[6] * x[10], x[3] * x[7] * x[11],
    //   ...
    // ].
    // If n is not a multiple of WIDTH, the result is truncated from the end. For example,
    // for n == 5, we get [x[0], x[1], x[2], x[3], x[0] * x[4]].
    let mut buf: Vec<F> = Vec::with_capacity(n);
    // cumul_prod holds the last WIDTH elements of buf. This is redundant, but it's how we
    // convince LLVM to keep the values in the registers.
    let mut cumul_prod: [F; WIDTH] = x[..WIDTH].try_into().unwrap();
    buf.extend(cumul_prod);
    for (i, &xi) in x[WIDTH..].iter().enumerate() {
        cumul_prod[i % WIDTH] *= xi;
        buf.push(cumul_prod[i % WIDTH]);
    }
    debug_assert_eq!(buf.len(), n);

    let mut a_inv = {
        // This is where the four dependency chains meet.
        // Take the last four elements of buf and invert them all.
        let c01 = cumul_prod[0] * cumul_prod[1];
        let c23 = cumul_prod[2] * cumul_prod[3];
        let c0123 = c01 * c23;
        let c0123inv = c0123.inv();
        let c01inv = c0123inv * c23;
        let c23inv = c0123inv * c01;
        [
            c01inv * cumul_prod[1],
            c01inv * cumul_prod[0],
            c23inv * cumul_prod[3],
            c23inv * cumul_prod[2],
        ]
    };

    for i in (WIDTH..n).rev() {
        // buf[i - WIDTH] has not been written to by this loop, so it equals
        // x[i % WIDTH] * x[i % WIDTH + WIDTH] * ... * x[i - WIDTH].
        buf[i] = buf[i - WIDTH] * a_inv[i % WIDTH];
        // buf[i] now holds the inverse of x[i].
        a_inv[i % WIDTH] *= x[i];
    }
    for i in (0..WIDTH).rev() {
        buf[i] = a_inv[i];
    }

    for (&bi, &xi) in buf.iter().zip(x) {
        // Sanity check only.
        debug_assert_eq!(bi * xi, F::one());
    }

    buf
}

/// p(x) -> p(x, y)
pub fn add_dummy_back<F: Field>(
    poly: &DenseMultilinearExtension<F>,
    num_vars_extended: usize,
) -> DenseMultilinearExtension<F> {
    let evaluations_extended: Vec<F> = poly
        .evaluations
        .iter()
        .cycle()
        .take(1 << (poly.num_vars + num_vars_extended))
        .cloned()
        .collect();
    DenseMultilinearExtension::from_vec(poly.num_vars + num_vars_extended, evaluations_extended)
}

/// p(y) -> p(x, y):= p(y)
pub fn add_dummy_front<F: Field>(
    poly: &DenseMultilinearExtension<F>,
    num_vars_extended: usize,
) -> DenseMultilinearExtension<F> {
    let evaluations_extended: Vec<F> = poly
        .evaluations
        .iter()
        .flat_map(|&x| std::iter::repeat(x).take(1 << num_vars_extended))
        .collect();
    DenseMultilinearExtension::from_vec(poly.num_vars + num_vars_extended, evaluations_extended)
}

pub fn absolute_range_tables<F: Field>(
    num_vars: usize,
    range: usize,
) -> Vec<Rc<DenseMultilinearExtension<F>>> {
    let mut acc = F::zero();
    let mut vec = Vec::with_capacity(2 * range - 1);
    (0..range).for_each(|i| {
        vec.push(acc);
        if i != 0 {
            vec.push(-acc);
        };
        acc += F::one();
    });
    //dbg!(acc);

    let poly_size = 1 << num_vars;
    let pad_size = (poly_size - (vec.len() % poly_size)) % poly_size;
    vec.resize(vec.len() + pad_size, F::zero());
    vec_to_polys(num_vars, &vec)
}

pub fn field_range_tables<F: Field>(
    num_vars: usize,
    min: F,
    max: F,
) -> Vec<Rc<DenseMultilinearExtension<F>>> {
    let mut acc = min;
    let mut vec = Vec::new();
    while acc != max + F::one() {
        vec.push(acc);
        acc += F::one();
    }

    let poly_size = 1 << num_vars;
    let pad_size = (poly_size - (vec.len() % poly_size)) % poly_size;
    vec.resize(vec.len() + pad_size, F::zero());
    vec_to_polys(num_vars, &vec)
}

pub fn range_tables<F: Field>(
    num_vars: usize,
    range: usize,
) -> Vec<Rc<DenseMultilinearExtension<F>>> {
    let mut vec = Vec::with_capacity(range);

    let mut acc = F::zero();
    (0..range).for_each(|_| {
        vec.push(acc);
        acc += F::one();
    });

    let poly_size = 1 << num_vars;
    let pad_size = (poly_size - (vec.len() % poly_size)) % poly_size;
    vec.resize(vec.len() + pad_size, F::zero());
    vec_to_polys(num_vars, &vec)
}

pub fn function_table<F: Field>(
    num_vars: usize,
    domain: &[F],
    func: fn(F) -> F,
) -> (
    Vec<Rc<DenseMultilinearExtension<F>>>,
    Vec<Rc<DenseMultilinearExtension<F>>>,
) {
    let x = domain.to_vec();
    let y: Vec<F> = domain.iter().map(|&x| func(x)).collect();
    dbg!(&y);
    let tx_vec = x
        .chunks_exact(1 << num_vars)
        .map(|evals| Rc::new(DenseMultilinearExtension::from_slice(num_vars, evals)))
        .collect();
    let ty_vec = y
        .chunks_exact(1 << num_vars)
        .map(|evals| Rc::new(DenseMultilinearExtension::from_slice(num_vars, evals)))
        .collect();
    (tx_vec, ty_vec)
}

pub fn concat_slices<F: Clone>(slices: &[&[F]]) -> Vec<F> {
    let total_len: usize = slices.iter().map(|s| s.len()).sum();
    let mut combined = Vec::with_capacity(total_len);
    for &slice in slices {
        combined.extend_from_slice(slice);
    }
    combined
}

pub fn violent_multiplicity<F: Field>(
    f_vec: &[Rc<DenseMultilinearExtension<F>>],
    t_vec: &[Rc<DenseMultilinearExtension<F>>],
) -> Vec<Rc<DenseMultilinearExtension<F>>> {
    let num_vars = f_vec[0].num_vars;
    let t_iter = t_vec.iter().flat_map(|t| t.evaluations.iter());
    let m_eval: Vec<F> = t_iter
        .map(|t_item| {
            let m_f_vec = f_vec.iter().fold(F::zero(), |acc, f| {
                let m_f: usize = f
                    .evaluations
                    .iter()
                    .filter(|&f_item| f_item == t_item)
                    .count();
                let m_f: F = F::new(F::Value::as_from(m_f as f64));
                acc + m_f
            });

            let t_iter = t_vec.iter().flat_map(|t| t.evaluations.iter());
            let m_t = t_iter.filter(|&x| x == t_item).count();
            let m_t: F = F::new(F::Value::as_from(m_t as f64));
            m_f_vec / m_t
        })
        .collect();

    let m_vec: Vec<Rc<DenseMultilinearExtension<F>>> = m_eval
        .chunks_exact(1 << num_vars)
        .map(|eval| Rc::new(DenseMultilinearExtension::from_slice(num_vars, eval)))
        .collect();

    m_vec
}

pub fn multiplicity<F: PrimeField>(
    f_vec: &[Rc<DenseMultilinearExtension<F>>],
    t_vec: &[Rc<DenseMultilinearExtension<F>>],
) -> Vec<Rc<DenseMultilinearExtension<F>>> {
    let num_vars: usize = f_vec[0].num_vars;

    let f_vec: Vec<Vec<F>> = f_vec.iter().map(|f| reduce(&f.evaluations)).collect();
    let t_vec: Vec<Vec<F>> = t_vec.iter().map(|t| reduce(&t.evaluations)).collect();

    let mut f_table = HashMap::new();
    let mut t_table = HashMap::new();

    t_vec.iter().flat_map(|t| t.iter()).for_each(|&t_value| {
        t_table
            .entry(t_value)
            .and_modify(|counter| *counter += 1usize)
            .or_insert(1);
        f_table.insert(t_value, 0usize);
    });

    f_vec.iter().for_each(|f| {
        f.iter().for_each(|&f_value| {
            f_table.entry(f_value).and_modify(|counter| *counter += 1);
        })
    });

    let m_eval: Vec<F> = t_vec
        .iter()
        .flat_map(|t| t.iter())
        .map(|t_value| {
            let m_f = f_table[t_value];
            let m_t = t_table[t_value];
            let m_f = F::new(F::Value::as_from(m_f as u64));
            let m_t = F::new(F::Value::as_from(m_t as u64));
            m_f / m_t
        })
        .collect();

    let m_vec: Vec<Rc<DenseMultilinearExtension<F>>> = m_eval
        .chunks_exact(1 << num_vars)
        .map(|eval| Rc::new(DenseMultilinearExtension::from_slice(num_vars, eval)))
        .collect();
    debug_assert!(m_vec.len() == t_vec.len());
    m_vec
}

pub fn xy_multiplicity<F: PrimeField>(
    fx_vec: &[Rc<DenseMultilinearExtension<F>>],
    fy_vec: &[Rc<DenseMultilinearExtension<F>>],
    tx_vec: &[Rc<DenseMultilinearExtension<F>>],
    ty_vec: &[Rc<DenseMultilinearExtension<F>>],
) -> Vec<Rc<DenseMultilinearExtension<F>>> {
    let num_vars: usize = fx_vec[0].num_vars;
    assert!(fx_vec.len() == fy_vec.len());
    assert!(tx_vec.len() == ty_vec.len());

    let fx_vec: Vec<Vec<F>> = fx_vec.iter().map(|f| reduce(&f.evaluations)).collect();
    let fy_vec: Vec<Vec<F>> = fy_vec.iter().map(|f| reduce(&f.evaluations)).collect();
    let tx_vec: Vec<Vec<F>> = tx_vec.iter().map(|t| reduce(&t.evaluations)).collect();
    let ty_vec: Vec<Vec<F>> = ty_vec.iter().map(|t| reduce(&t.evaluations)).collect();

    let mut f_table = HashMap::new();
    let mut t_table = HashMap::new();

    tx_vec
        .iter()
        .flat_map(|tx| tx.iter())
        .zip(ty_vec.iter().flat_map(|ty| ty.iter()))
        .for_each(|(&tx, &ty)| {
            t_table
                .entry((tx, ty))
                .and_modify(|counter| *counter += 1usize)
                .or_insert(1);
            f_table.insert((tx, ty), 0usize);
        });

    fx_vec
        .iter()
        .flat_map(|fx| fx.iter())
        .zip(fy_vec.iter().flat_map(|fy| fy.iter()))
        .for_each(|(&fx, &fy)| {
            f_table.entry((fx, fy)).and_modify(|counter| *counter += 1);
        });

    let m_eval: Vec<F> = tx_vec
        .iter()
        .flat_map(|tx| tx.iter())
        .zip(ty_vec.iter().flat_map(|ty| ty.iter()))
        .map(|(&tx, &ty)| {
            let m_f = f_table[&(tx, ty)];
            let m_t = t_table[&(tx, ty)];
            let m_f = F::new(F::Value::as_from(m_f as u64));
            let m_t = F::new(F::Value::as_from(m_t as u64));
            m_f / m_t
        })
        .collect();

    let m_vec: Vec<Rc<DenseMultilinearExtension<F>>> = vec_to_polys(num_vars, &m_eval);

    debug_assert!(m_vec.len() == tx_vec.len());
    debug_assert!(m_vec.len() == ty_vec.len());
    m_vec
}

/// Multiplies a matrix `w` (stored in column-major order) by a column vector `x`.
/// Returns a new vector representing the resulting column vector.
///
/// `w`: Column-major stored matrix with dimensions m x n (m rows, n columns)
/// `x`: Column vector of length n
pub fn mat_vec_mul_column_major<F>(w: &[F], x: &[F]) -> Vec<F>
where
    F: std::ops::AddAssign + std::ops::Mul<Output = F> + Default + Clone,
{
    let n = x.len(); // Number of columns in w (and length of x)
    let m = w.len() / n; // Number of rows in w

    let mut result = vec![F::default(); m];

    for col in 0..n {
        for row in 0..m {
            result[row] += w[col * m + row].clone() * x[col].clone();
        }
    }

    result
}

pub fn compute_wt_w<F: Field>(w: &[F], num_row: usize, num_col: usize) -> Vec<F> {
    let mut wt_w = vec![F::zero(); num_col * num_col];
    // 遍历 W 的列并计算 W^T * W
    for i in 0..num_col {
        for j in i..num_col {
            let mut sum = F::zero();
            for k in 0..num_row {
                let w_ik = w[i * num_row + k]; // W 的第 i 列，第 k 行元素
                let w_jk = w[j * num_row + k]; // W 的第 j 列，第 k 行元素
                sum += w_ik * w_jk;
            }
            wt_w[i * num_col + j] = sum;
            wt_w[j * num_col + i] = sum; // 对称赋值
        }
    }
    wt_w
}

pub fn reduce<F: PrimeField>(vec: &[F]) -> Vec<F> {
    vec.iter()
        .map(|x| {
            if x.value() > F::MODULUS_VALUE {
                F::new(x.value() - F::MODULUS_VALUE)
            } else {
                *x
            }
        })
        .collect()
}

pub fn sample_from_intervals<R: Rng>(rng: &mut R, x: usize, q: usize) -> usize {
    assert!(x > 0 && q > x, "Invalid range parameters");

    // 计算从第一个区间采样的概率
    let probability = (x as f64) / (q as f64);

    // 判断采样是从哪个区间
    if rng.gen_bool(probability) {
        // 从区间 (0, x) 采样
        rng.gen_range(0..x)
    } else {
        // 从区间 (q-x, q-1) 采样
        rng.gen_range(q - x..q)
    }
}

#[cfg(test)]
mod test {
    use crate::utils::{eval_identity_poly, gen_identity_poly};
    use algebra::{
        //derive::{Field, Prime},
        BabyBear,
        FieldUniformSampler,
        MultilinearExtension,
    };
    use rand::thread_rng;
    use rand_distr::Distribution;

    // #[derive(Field, Prime)]
    // #[modulus = 132120577]
    // pub struct Fp32(u32);
    // field type
    type FF = BabyBear;

    #[test]
    fn test_gen_identity_evaluations() {
        let sampler = <FieldUniformSampler<FF>>::new();
        let mut rng = thread_rng();
        let dim = 10;
        let u: Vec<_> = (0..dim).map(|_| sampler.sample(&mut rng)).collect();

        let identity_at_u = gen_identity_poly(&u);

        let v: Vec<_> = (0..dim).map(|_| sampler.sample(&mut rng)).collect();

        assert_eq!(eval_identity_poly(&u, &v), identity_at_u.evaluate(&v));
    }
}

pub fn hashmap_memory_size<K, V>(map: &HashMap<K, V>) -> usize {
    let mut total_size = mem::size_of_val(map);

    for (key, value) in map.iter() {
        total_size += mem::size_of_val(key);
        total_size += mem::size_of_val(value);
    }
    total_size
}

// i 2f -> i f
#[inline]
pub fn relu_i128(x: i128, i: usize, f: usize) -> i128 {
    if x > 0 && x < (1 << (i + 2 * f)) {
        x >> f
    } else {
        0
    }
}

// i 2f -> i f
pub fn relu<F: PrimeField>(x: F, i: usize, f: usize) -> F {
    let x: u64 = x.value().into();
    let p: u64 = F::MODULUS_VALUE.into();

    if x < p / 2 && x < (1 << (i + 2 * f)) {
        F::new(F::Value::as_from(x >> f))
    } else {
        F::zero()
    }
}

pub fn is_valid<F: PrimeField>(x: F, i_bit: usize, f_bit: usize) -> bool {
    let mut x: u64 = x.value().into();
    let p: u64 = F::MODULUS_VALUE.into();
    if x > p {
        x -= p;
    }

    let result = if x < p / 2 {
        x < (1 << (i_bit + f_bit))
    } else {
        (p - x) < (1 << (i_bit + f_bit))
    };

    if !result {
        println!("x: {}, valid range {}", x, 1 << (i_bit + f_bit));
    };

    result
}

#[inline]
pub fn compute_a_b<F: Field>(
    a: &[F],
    b: &[F],
    num_l: usize, // A 的行数
    num_m: usize, // A 的列数 (等于 B 的行数)
    num_r: usize, // B 的列数
) -> Vec<F> {
    // 确保矩阵 A 的元素数量与提供的维度一致
    assert!(
        a.len() == num_l * num_m,
        "矩阵 A 的大小与提供的维度不匹配！"
    );
    // 确保矩阵 B 的元素数量与提供的维度一致
    assert!(
        b.len() == num_m * num_r,
        "矩阵 B 的大小与提供的维度不匹配！"
    );

    // 初始化结果矩阵 C，大小为 num_l x num_r，初始值为零
    let mut c = vec![F::zero(); num_l * num_r];

    // 遍历 B 的每一列
    for j in 0..num_r {
        // 遍历 A 的每一行
        for i in 0..num_l {
            let mut sum = F::zero();
            // 计算 C[i, j] = sum_k A[i, k] * B[k, j]
            for k in 0..num_m {
                // 在列主序中，A[i, k] 位于 a[k * num_l + i]
                let a_ik = a[k * num_l + i];
                // 在列主序中，B[k, j] 位于 b[j * num_m + k]
                let b_kj = b[j * num_m + k];
                sum = sum + (a_ik * b_kj);
            }
            // 在列主序中，C[i, j] 位于 c[j * num_l + i]
            c[j * num_l + i] = sum;
        }
    }

    c
}

pub fn is_valid_i128(x: i128, i_bit: usize, f_bit: usize) -> bool {
    let result = (-(1 << (i_bit + f_bit)) < x) && (x < (1 << (i_bit + f_bit)));
    if !result {
        println!("x: {}, valid range {}", x, 1 << (i_bit + f_bit));
    };

    result
}



pub fn estimate_spectral_norm(matrix: &DMatrix<f64>, n_iter: usize) -> f64 {
    let cols = matrix.ncols();
    let mut u = DVector::<f64>::from_element(cols, 1.0);
    u /= u.norm();

    for _ in 0..n_iter {
        let v = matrix.transpose() * &u;
        let v = &v / v.norm();
        u = matrix * &v;
        u /= u.norm();
    }

    let v = matrix.transpose() * &u;
    let spectral_norm_sq = (u.transpose() * matrix * &v)[(0, 0)];
    spectral_norm_sq.sqrt()
}

pub fn spectral_normalize(matrix: &DMatrix<f64>, upper_bound: f64) -> DMatrix<f64> {
    let sigma = estimate_spectral_norm(matrix, 5); // 可调迭代次数
    if sigma <= upper_bound {
        matrix.clone()
    } else {
        matrix * (upper_bound / sigma)
    }
}

pub fn random_normalized_w<F: PrimeField>(
    name: &String,
    num_row: usize,
    num_col: usize,
    i_bit: usize,
    f_bit: usize,
    spec_norm: f64,
) -> Rc<DenseMultilinearExtension<F>> {
    let range = 1 << (f_bit);
    let mut rng = rand::thread_rng();
    let w = DMatrix::from_fn(num_row, num_col, |_, _| {
        rng.gen_range(-(range as i64) + 1..(range as i64)) as f64
    });
    let w = spectral_normalize(&w, spec_norm);

    let w = w.map(|x| (x as f64 * (1 << f_bit) as f64).round() as i128);
    w.iter()
        .for_each(|x| assert!(is_valid_i128(*x, i_bit, f_bit)));
    let w: Vec<i128> = w.iter().cloned().collect();
    let w = vec_to_named_poly(name, &i128_to_field(&w));
    w
}
