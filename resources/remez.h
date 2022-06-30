#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <atomic>
#include <iostream>
#include <random>
#include <signal.h>
#include <unistd.h>
#include <mpfr.h>

struct Float {
public:
    Float() { memset(value, 0, sizeof(__mpfr_struct)); }

    template <typename T, std::enable_if_t<std::is_scalar_v<T>, int> = 0>
    Float(const T &v) {
        mpfr_init(value);
        if constexpr (std::is_same_v<T, long double>)
            mpfr_set_ld(value, v, MPFR_RNDN);
        else if constexpr (std::is_same_v<T, double>)
            mpfr_set_d(value, v, MPFR_RNDN);
        else if constexpr (std::is_same_v<T, float>)
            mpfr_set_flt(value, v, MPFR_RNDN);
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
            mpfr_set_sj(value, (intmax_t) v, MPFR_RNDN);
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>)
            mpfr_set_uj(value, (uintmax_t) v, MPFR_RNDN);
        else
            throw std::runtime_error("Float::Float(): unsupported type!");
    }

    Float(const Float &f) {
        mpfr_init_set(value, f.value, MPFR_RNDN);
    }

    ~Float() {
        if (value[0]._mpfr_d)
            mpfr_clear(value);
    }

    Float(Float &&f) {
        memcpy(value, f.value, sizeof(__mpfr_struct));
        memset(f.value, 0, sizeof(__mpfr_struct));
    }

    Float &operator=(Float &&f) {
        if (value[0]._mpfr_d)
            mpfr_clear(value);
        memcpy(value, f.value, sizeof(__mpfr_struct));
        memset(f.value, 0, sizeof(__mpfr_struct));
        return *this;
    }

    Float &operator=(const Float &f) {
        if (value[0]._mpfr_d == nullptr)
            mpfr_init(value);
        mpfr_set(value, f.value, MPFR_RNDN);
        return *this;
    }

    Float operator+(const Float &f) const {
        Float result;
        mpfr_init(result.value);
        mpfr_add(result.value, value, f.value, MPFR_RNDN);
        return result;
    }

    Float &operator+=(const Float &f) {
        mpfr_add(value, value, f.value, MPFR_RNDN);
        return *this;
    }

    template <typename T>
    friend Float operator+(const T &v, const Float &f) {
        return Float(v) + f;
    }

    Float operator-(const Float &f) const {
        Float result;
        mpfr_init(result.value);
        mpfr_sub(result.value, value, f.value, MPFR_RNDN);
        return result;
    }

    Float &operator-=(const Float &f) {
        mpfr_sub(value, value, f.value, MPFR_RNDN);
        return *this;
    }

    template <typename T>
    friend Float operator-(const T &v, const Float &f) {
        return Float(v) - f;
    }

    Float operator*(const Float &f) const {
        Float result;
        mpfr_init(result.value);
        mpfr_mul(result.value, value, f.value, MPFR_RNDN);
        return result;
    }

    Float &operator*=(const Float &f) {
        mpfr_mul(value, value, f.value, MPFR_RNDN);
        return *this;
    }

    template <typename T>
    friend Float operator*(const T &v, const Float &f) {
        return Float(v) * f;
    }

    Float operator/(const Float &f) const {
        Float result;
        mpfr_init(result.value);
        mpfr_div(result.value, value, f.value, MPFR_RNDN);
        return result;
    }

    Float &operator/=(const Float &f) {
        mpfr_div(value, value, f.value, MPFR_RNDN);
        return *this;
    }

    template <typename T>
    friend Float operator/(const T &v, const Float &f) {
        return Float(v) / f;
    }

    bool operator<(const Float &f) const {
        return mpfr_less_p(value, f.value);
    }

    bool operator<=(const Float &f) {
        return mpfr_lessequal_p(value, f.value);
    }

    bool operator>(const Float &f) const {
        return mpfr_greater_p(value, f.value);
    }

    bool operator>=(const Float &f) const {
        return mpfr_greaterequal_p(value, f.value);
    }

    bool operator==(const Float &f) const {
        return mpfr_equal_p(value, f.value);
    }

    bool operator!=(const Float &f) const {
        return !mpfr_equal_p(value, f.value);
    }

    Float operator-() const {
        Float result;
        mpfr_init(result.value);
        mpfr_neg(result.value, value, MPFR_RNDN);
        return result;
    }

    template <typename T> T cast() const {
        if constexpr (std::is_same_v<T, long double>)
            return mpfr_get_ld(value, MPFR_RNDN);
        else if constexpr (std::is_same_v<T, double>)
            return mpfr_get_d(value, MPFR_RNDN);
        else if constexpr (std::is_same_v<T, float>)
            return mpfr_get_flt(value, MPFR_RNDN);
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
            return (T) mpfr_get_sj(value, MPFR_RNDN);
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>)
            return (T) mpfr_get_uj(value, MPFR_RNDN);
        else
            throw std::runtime_error("Float::cast(): unsupported type!");
    }

    static Float pi() {
        Float result;
        mpfr_init(result.value);
        mpfr_const_pi(result.value, MPFR_RNDN);
        return result;
    }

    static Float inf() {
        Float result;
        mpfr_init(result.value);
        mpfr_inf_p(result.value);
        return result;
    }

    void print_mathematica() {
        const char *fmt = "%Re";
        char suffix[100];
        snprintf(suffix, sizeof(suffix), "`%i*10^",
                 (int) ceil(mpfr_get_prec(value) * log(2) / log(10)) + 1);
        size_t suffix_size = strlen(suffix);
        size_t size = mpfr_snprintf(nullptr, 0, fmt, value) + suffix_size + 1;
        char *buf = (char *) malloc(size);
        mpfr_snprintf(buf, size, fmt, value);
        char *exp = strchr(buf, 'e');
        if (exp) {
            memmove(exp + suffix_size, exp + 1, strlen(exp) + 1);
            memcpy(exp, suffix, suffix_size);
        }
        fputs(buf, stdout);
        free(buf);
    }

    mpfr_t value;
};

inline Float fma_(const Float &a, const Float &b, const Float &c) {
    Float result;
    mpfr_init(result.value);
    mpfr_fma(result.value, a.value, b.value, c.value, MPFR_RNDN);
    return result;
}

inline float fma_(const float &a, const float &b, const float &c) {
    return std::fma(a, b, c);
}

inline double fma_(const double &a, const double &b, const double &c) {
    return std::fma(a, b, c);
}


inline long double abs_(const long double &a) { return std::abs(a); }

inline bool isnan_(const float &a) { return std::isnan(a); }
inline bool isnan_(const double &a) { return std::isnan(a); }
inline bool isnan_(const Float &a) { return (bool) mpfr_nan_p(a.value); }

inline bool isinf_(const float &a) { return std::isinf(a); }
inline bool isinf_(const double &a) { return std::isinf(a); }
inline bool isinf_(const Float &a) { return (bool) mpfr_inf_p(a.value); }

#define WRAP1(name)                                                            \
    inline Float name##_(const Float &a) {                                     \
        Float result;                                                          \
        mpfr_init(result.value);                                               \
        mpfr_##name(result.value, a.value, MPFR_RNDN);                         \
        return result;                                                         \
    }                                                                          \
    inline float name##_(const float &a) {                                     \
        return std::name(a);                                                   \
    }                                                                          \
    inline double name##_(const double &a) {                                   \
        return std::name(a);                                                   \
    }


#define WRAP2(name)                                                            \
    inline Float name##_(const Float &a, const Float &b) {                     \
        Float result;                                                          \
        mpfr_init(result.value);                                               \
        mpfr_##name(result.value, a.value, b.value, MPFR_RNDN);                \
        return result;                                                         \
    }                                                                          \
    inline float name##_(const float &a, const float &b) {                     \
        return std::name(a, b);                                                \
    }                                                                          \
    inline double name##_(const double &a, const double &b) {                  \
        return std::name(a, b);                                                \
    }

WRAP1(abs)
WRAP1(sqrt)
WRAP1(exp)
WRAP1(exp2)
WRAP1(log)
WRAP1(log2)
WRAP1(erf)
WRAP1(erfc)
WRAP1(sin)
WRAP1(cos)
WRAP1(tan)
WRAP1(asin)
WRAP1(acos)
WRAP1(atan)
WRAP1(sinh)
WRAP1(cosh)
WRAP1(tanh)
WRAP1(asinh)
WRAP1(acosh)
WRAP1(atanh)
WRAP2(atan2)
WRAP2(pow)
WRAP2(min)
WRAP2(max)

#undef WRAP1
#undef WRAP2

static bool stop = false;

// Horner-style evaluation of a polynomial of degree 'n'
template <typename Value> Value horner(Value x, const Value *coeffs, int n) {
    Value accum = coeffs[n];
    for (int i = 1; i <= n; ++i)
        accum = fma_(x, accum, coeffs[n - i]);
    return accum;
}

// Estrin-style evaluation of a polynomial of degree 'n'
template <typename Value> Value estrin(Value x, const Value *coeffs, int n) {
    int n_rec = n / 2, n_fma = (n + 1) / 2;

    Value *coeffs_rec = (Value *) alloca(sizeof(Value) * (n_rec + 1));

    for (int i = 0; i < n_fma; ++i)
        coeffs_rec[i] = fma_(x, coeffs[2 * i + 1], coeffs[2 * i]);

    if (n_rec == n_fma)
        coeffs_rec[n_rec] = coeffs[n];

    if (n_rec == 0)
        return coeffs_rec[0];
    else
        return estrin(x * x, coeffs_rec, n_rec);
}

template <size_t n, typename Value> __attribute__ ((always_inline))
inline Value estrin_static(const Value &x, const Value *coeffs) {
    constexpr size_t n_rec = (n - 1) / 2, n_fma = n / 2;

    Value coeffs_rec[n_rec + 1];
    #pragma unroll
    for (size_t i = 0; i < n_fma; ++i)
        coeffs_rec[i] = fma_(x, coeffs[2 * i + 1], coeffs[2 * i]);

    if constexpr (n_rec == n_fma) // odd case
        coeffs_rec[n_rec] = coeffs[n - 1];

    if constexpr (n_rec == 0)
        return coeffs_rec[0];
    else
        return estrin_static<n_rec + 1>(x * x, coeffs_rec);
}

template <typename Value>
inline Value estrin_fast(const Value &x, const Value *coeffs, size_t n) {
    switch (n) {
        case 0:  return estrin_static<1>  (x, coeffs);
        case 1:  return estrin_static<2>  (x, coeffs);
        case 2:  return estrin_static<3>  (x, coeffs);
        case 3:  return estrin_static<4>  (x, coeffs);
        case 4:  return estrin_static<5>  (x, coeffs);
        case 5:  return estrin_static<6>  (x, coeffs);
        case 6:  return estrin_static<7>  (x, coeffs);
        case 7:  return estrin_static<8>  (x, coeffs);
        case 8:  return estrin_static<9>  (x, coeffs);
        case 9:  return estrin_static<10> (x, coeffs);
        case 10: return estrin_static<11> (x, coeffs);
        case 11: return estrin_static<12> (x, coeffs);
        case 12: return estrin_static<13> (x, coeffs);
        case 13: return estrin_static<14> (x, coeffs);
        case 14: return estrin_static<15> (x, coeffs);
        case 15: return estrin_static<16> (x, coeffs);
        case 16: return estrin_static<17> (x, coeffs);
        case 17: return estrin_static<18> (x, coeffs);
        case 18: return estrin_static<19> (x, coeffs);
        case 19: return estrin_static<20> (x, coeffs);
        case 20: return estrin_static<21> (x, coeffs);
        case 21: return estrin_static<22> (x, coeffs);
        case 22: return estrin_static<23> (x, coeffs);
        case 23: return estrin_static<24> (x, coeffs);
        case 24: return estrin_static<25> (x, coeffs);
        case 25: return estrin_static<26> (x, coeffs);
        case 26: return estrin_static<27> (x, coeffs);
        case 27: return estrin_static<28> (x, coeffs);
        case 28: return estrin_static<29> (x, coeffs);
        case 29: return estrin_static<30> (x, coeffs);
        default: return estrin(x, coeffs, n);
    }
}

template <typename Func>
Float golden(Float a, Float b, const Func &f) {
    const Float invphi        = (sqrt_(Float(5)) - 1) / 2,
                invphi2       = invphi*invphi,
                tol           = 1e-5;

    Float h = b - a;
    int n_steps = (log_(tol / h) / log_(invphi)).cast<int>();

    Float c = a + invphi2 * h,
           d = a + invphi * h,
           fc = f(c),
           fd = f(d);

    for (int i = 0; i < n_steps; ++i) {
        if (fc < fd) {
            b = d; d = c; fd = fc;
            h = invphi * h;
            c = fma_(h, invphi2, a);
            fc = f(c);
        } else {
            a = c; c = d; fc = fd;
            h = invphi * h;
            d = fma_(h, invphi, a);
            fd = f(d);
        }
    }

    return .5 * ((fc < fd) ? (a + d) : (c + b));
}

// Variant on the false position method, finds a root on [a, b]
template <typename Func> Float illinois(Float a, Float b, const Func &f) {
    Float fa = f(a), fb = f(b);

    if (!(fa * fb < 0)) {
        mpfr_fprintf(stderr,
                     "illinos(%.5Re, %.5Re): called with a non-bracketing "
                     "interval (%.5Re, %.5Re)!\n",
                     a.value, b.value, fa.value, fb.value);
        return Float(std::numeric_limits<float>::quiet_NaN());
    }

    while (true) {
        Float c = b - fb * (b - a) / (fb - fa);

        if (abs_(c - b) < 1e-8)
            return c;

        Float fc = f(c);

        if (fc * fb < 0) {
            a = b;
            fa = fb;
        } else {
            fa *= .5f;
        }
        b = c;
        fb = fc;
    }
}

/// LU factorization with partial pivoting
bool lu(int n, Float *A, int *pivot) {
    for (int i = 0; i < n; i++)
        pivot[i] = i;

    for (int i = 0; i < n; i++) {
        Float maxval = 0.0;
        int imax = i;

        for (int k = i; k < n; k++) {
            Float a = abs_(A[k*n + i]);
            if (a > maxval) {
                maxval = a;
                imax = k;
            }
        }

        if (maxval == 0)
            return false;

        if (imax != i) {
            int j = pivot[i];
            pivot[i] = pivot[imax];
            pivot[imax] = j;

            for (int j = 0; j < n; ++j) {
                Float tmp = A[i*n + j];
                A[i*n + j] = A[imax*n + j];
                A[imax*n + j] = tmp;
            }
        }

        for (int j = i + 1; j < n; j++) {
            A[j*n + i] /= A[i*n + i];

            for (int k = i + 1; k < n; k++)
                A[j*n + k] = fma_(-A[j*n + i], A[i*n + k], A[j*n + k]);
        }
    }

    return true;
}

/// Forward and back-substitute a solution vector through a LU factorizatoin
void lu_solve(int n, Float *A, Float *b, Float *x, int *pivot) {
    for (int i = 0; i < n; i++) {
        x[i] = b[pivot[i]];

        for (int k = 0; k < i; k++)
            x[i] = fma_(-A[i*n+k], x[k], x[i]);
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int k = i + 1; k < n; k++)
            x[i] = fma_(-A[i*n+k], x[k], x[i]);

        x[i] = x[i] / A[i*n+i];
    }
}

/// Reinterpret the binary represesentation of a data type
template<typename Target, typename Source> Target memcpy_cast(const Source &source) {
    static_assert(sizeof(Source) == sizeof(Target), "memcpy_cast: sizes did not match!");
    Target target;
    std::memcpy(&target, &source, sizeof(Target));
    return target;
}

/**
 * \brief Atomic floating point data type
 *
 * The class implements an an atomic floating point data type (which is not
 * possible with the existing overloads provided by <tt>std::atomic</tt>). It
 * internally casts floating point values to an integer storage format and uses
 * atomic integer compare and exchange operations to perform changes.
 */
template <typename Type = float> class AtomicFloat {
private:
    using Storage = std::conditional_t<sizeof(Type) == 4, uint32_t, uint64_t>;

public:
    /// Initialize the AtomicFloat with a given floating point value
    explicit AtomicFloat(Type v = 0.f) { m_bits = memcpy_cast<Storage>(v); }

    /// Convert the AtomicFloat into a normal floating point value
    operator Type() const { return memcpy_cast<Type>(m_bits.load(std::memory_order_relaxed)); }

    /// Overwrite the AtomicFloat with a floating point value
    AtomicFloat &operator=(Type v) { m_bits = memcpy_cast<Storage>(v); return *this; }

    /// Atomically add a floating point value
    AtomicFloat &operator+=(Type arg) { return do_atomic([arg](Type value) { return value + arg; }); }

    /// Atomically subtract a floating point value
    AtomicFloat &operator-=(Type arg) { return do_atomic([arg](Type value) { return value - arg; }); }

    /// Atomically multiply by a floating point value
    AtomicFloat &operator*=(Type arg) { return do_atomic([arg](Type value) { return value * arg; }); }

    /// Atomically divide by a floating point value
    AtomicFloat &operator/=(Type arg) { return do_atomic([arg](Type value) { return value / arg; }); }

    /// Atomically compute the minimum
    AtomicFloat &min(Type arg) { return do_atomic([arg](Type value) { return std::min(value, arg); }); }

    /// Atomically compute the maximum
    AtomicFloat &max(Type arg) { return do_atomic([arg](Type value) { return std::max(value, arg); }); }

protected:
    /// Apply a FP operation atomically (verified that this will be nicely inlined in the above operators)
    template <typename Func> AtomicFloat& do_atomic(Func func) {
        Storage old_bits = m_bits.load(std::memory_order::memory_order_relaxed), new_bits;
        do {
            new_bits = memcpy_cast<Storage>(func(memcpy_cast<Type>(old_bits)));
            if (new_bits == old_bits)
                break;
        } while (!m_bits.compare_exchange_weak(old_bits, new_bits));
        return *this;
    }

protected:
    std::atomic<Storage> m_bits;
};

template <typename Func, typename Target> struct Annealer {
    using Int = std::conditional_t<sizeof(Target) == 4, int32_t, int64_t>;

    /// Degree of polynomial in numerator
    int deg_p;

    /// Degree of polynomial in denominator
    int deg_q;

    /// Target function
    Func func;

    /// Interval
    Target start, end;

    /// Polynomial coefficients
    std::unique_ptr<Target[]> coeffs_cur;
    std::unique_ptr<Target[]> coeffs_prop;
    std::unique_ptr<Target[]> coeffs_best;

    /// Error
    std::pair<float, float> err_cur, err_prop, err_best;

    /// Annealing parameters
    size_t sample_count;
    size_t iterations;
    size_t cycles;
    bool estrin;

    /// Reference values
    std::unique_ptr<Target[]> x;
    std::unique_ptr<long double[]> x_ulp;
    std::unique_ptr<long double[]> y;

    Annealer(int deg_p, int deg_q, Func func, Float start, Float end,
             Float *coeffs_, size_t sample_count, size_t iterations,
             size_t cycles, bool estrin)
        : deg_p(deg_p), deg_q(deg_q), func(func),
          start(start.cast<Target>()), end(end.cast<Target>()),
          coeffs_cur(new Target[deg_p + deg_q + 2]),
          coeffs_prop(new Target[deg_p + deg_q + 2]),
          coeffs_best(new Target[deg_p + deg_q + 2]),
          sample_count(sample_count),
          iterations(iterations),
          cycles(cycles),
          estrin(estrin),
          x(new Target[sample_count]),
          x_ulp(new long double[sample_count]),
          y(new long double[sample_count]) {

        fflush(stdout);
        for (int i = 0; i < deg_p + deg_q + 2; ++i)
            coeffs_cur[i] = coeffs_[i].cast<Target>();
        coeffs_prop[deg_p + 1] = 1;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < sample_count; ++i) {
            Float xf = start + (end - start) * Float(i) / Float(sample_count - 1);
            Target xt = xf.cast<Target>();
            Float yf = func(Float(xt));
            Target yt = yf.cast<Target>();

            x[i] = xt;
            y[i] = yf.cast<long double>();
            x_ulp[i] = (long double) std::nextafter(yt, std::numeric_limits<Target>::infinity()) - yt;
        }
        err_cur = err_best = error(coeffs_cur.get());
        memcpy(coeffs_best.get(), coeffs_cur.get(),
               sizeof(Target) * (deg_p + deg_q + 2));
    }

    void dump() {
        const char *fmt = sizeof(float) == 4 ? "        %s%a%s // %s%.9e\n"
                                             : "        %s%a%s // %s%.17e\n";
        printf("\n   p = %s(x,\n", estrin ? "estrin" : "horner");
        for (int i = 0; i <= deg_p; ++i)
            printf(fmt,
                   coeffs_best[i] >= 0 ? " " : "", coeffs_best[i],
                   i < deg_p ? "," : " ",
                   coeffs_best[i] >= 0 ? " " : "", coeffs_best[i]);
        if (deg_q > 0) {
            printf("   );\n   q = %s(x,\n", estrin ? "estrin" : "horner");
            for (int i = 0; i <= deg_q; ++i)
                printf(fmt,
                       coeffs_best[i + deg_p + 1] >= 0 ? " " : "", coeffs_best[i + deg_p + 1],
                       i < deg_q ? "," : "",
                       coeffs_best[i + deg_p + 1] >= 0 ? " " : "", coeffs_best[i + deg_p + 1]);
        }
        printf("   );\n\n");

        printf("   Restart search with -C ");
        for (int i = 0; i < deg_p + deg_q + 2; ++i)
            printf("%a%s", coeffs_best[i], (i < deg_p + deg_q + 1) ? "," : "");
        printf("\n\n");
    }

    std::pair<float, float> error(Target *c) const {
        double err_sum = 0;
        AtomicFloat<double> err_max(0);

        if (estrin) {
            #pragma omp parallel for schedule(static) reduction(+:err_sum)
            for (size_t i = 0; i < sample_count; ++i) {
                Target xt    = x[i],
                       num   = ::estrin_fast(xt, c, deg_p),
                       denom = ::estrin_fast(xt, c + deg_p + 1, deg_q),
                       value = num / denom;

                double err = (double) (abs_(value - y[i]) / x_ulp[i]);

                err_max.max(err);
                err_sum += err;
            }
        } else {
            #pragma omp parallel for schedule(static) reduction(+:err_sum)
            for (size_t i = 0; i < sample_count; ++i) {
                Target xt    = x[i],
                       num   = ::horner(xt, c, deg_p),
                       denom = ::horner(xt, c + deg_p + 1, deg_q),
                       value = num / denom;

                double err = (double) (abs_(value - y[i]) / x_ulp[i]);

                err_max.max(err);
                err_sum += err;
            }
        }

        return { (float) err_max, (float) (err_sum / sample_count) };
    }

    void check() {
        if (start * end >= 0) {
            check(memcpy_cast<Int>(start), memcpy_cast<Int>(end));
        } else {
            check(memcpy_cast<Int>((Target) 0.f), memcpy_cast<Int>(start));
            check(memcpy_cast<Int>(end), memcpy_cast<Int>((Target) -0.f));
        }
    }

    void check(Int start_i, Int end_i) {
        printf("Brute force accuracy check from %f to %f ..\n",
                memcpy_cast<Target>(start_i),
                memcpy_cast<Target>(end_i));

        if (start_i >= end_i) {
            printf("Annealer::check(): internal error!\n");
            exit(-1);
        }

        double err_sum = 0;
        AtomicFloat<double> err_max(0);

        if (estrin) {
            #pragma omp parallel for schedule(static) reduction(+:err_sum)
            for (Int i = start_i; i < end_i; ++i) {
                if (stop)
                    continue;

                Target xt    = memcpy_cast<Target>(i),
                       num   = ::estrin_fast(xt, coeffs_best.get(), deg_p),
                       denom = ::estrin_fast(xt, coeffs_best.get() + deg_p + 1, deg_q),
                       value = num / denom,
                       ulp   = std::nextafter(value, std::numeric_limits<Target>::infinity()) - value;

                double err = (abs_((Float) value - func((Float) xt)) / ulp).template cast<double>();

                err_max.max(err);
                err_sum += err;
            }
        } else {
            #pragma omp parallel for schedule(static) reduction(+:err_sum)
            for (Int i = start_i; i < end_i; ++i) {
                if (stop)
                    continue;

                Target xt    = memcpy_cast<Target>(i),
                       num   = ::horner(xt, coeffs_best.get(), deg_p),
                       denom = ::horner(xt, coeffs_best.get() + deg_p + 1, deg_q),
                       value = num / denom,
                       ulp   = std::nextafter(value, std::numeric_limits<Target>::infinity()) - value;

                double err = (abs_((Float) value - func((Float) xt)) / ulp).template cast<double>();

                err_max.max(err);
                err_sum += err;
            }
        }

        printf(" -> exhaustive search yields: max = %.2f ulp, avg = %.3f ulp.\n",
               (double) err_max, err_sum / (end_i - start_i));
    }

    void go() {
        std::mt19937 engine;
        std::uniform_real_distribution<float> uniform;
        std::uniform_int_distribution<uint32_t> uniform_i(0, deg_p + deg_q - 1);
        std::normal_distribution<float> normal;

        size_t n_invalid = 0, n_accepted = 0, n_rejected = 0;
        float scale = 1;

        printf("\n");
        for (int j = 0; j < cycles && !stop; ++j) {
            memcpy(coeffs_cur.get(), coeffs_best.get(),
                   sizeof(Target) * (deg_p + deg_q + 2));
            err_cur = err_best;
            for (size_t i = 0; i < iterations && !stop; ++i) {
                float temp = .5 * expf(i / -float(iterations - 1) * 5);

                if (i % 1000 == 0) {
                    printf("   %05zu: best max=%.3f ulp, best avg=%.3f ulp, cur max=%.3f ulp, cur avg=%.3f ulp, temp=%.1f.\n", i,
                           err_best.first, err_best.second,
                           err_cur.first, err_cur.second,
                           temp);
                    fflush(stdout);
                }

                int k = uniform_i(engine);
                if (k == deg_p + 1)
                    continue;
                Int value = memcpy_cast<Int>(coeffs_cur[k]);
                int shift = normal(engine) > 0 ? 1 : -1;
                value += shift;

                memcpy(coeffs_prop.get(), coeffs_cur.get(),
                       sizeof(Target) * (deg_p + deg_q + 2));
                coeffs_prop[k] = memcpy_cast<Target>(value);

                err_prop = error(coeffs_prop.get());

                float err_cur_w = err_cur.first * scale + err_cur.second,
                      err_prop_w = err_prop.first * scale + err_prop.second,
                      err_best_w = err_best.first * scale + err_best.second;

                if (err_cur_w < err_best_w) {
                    memcpy(coeffs_best.get(), coeffs_cur.get(),
                           sizeof(Target) * (deg_p + deg_q + 2));
                    err_best = err_cur;
                }

                if (err_prop_w < err_cur_w) {
                    coeffs_prop.swap(coeffs_cur);
                    err_prop.swap(err_cur);
                    n_accepted++;
                } else {
                    float sample = uniform(engine),
                          acceptance = std::exp((err_cur_w - err_prop_w) / temp);

                    if (sample < acceptance) {
                        coeffs_prop.swap(coeffs_cur);
                        err_prop.swap(err_cur);
                        n_accepted++;
                    } else {
                        n_rejected++;
                    }
                }
            }
        }
        printf("\n   -> %zu invalid, %zu accepted, %zu rejected steps.\n   ",
               n_invalid, n_accepted, n_rejected);
    }
};

template <typename Func> struct Remez {
    /// Degree of polynomial in numerator
    int deg_p;

    /// Degree of polynomial in denominator
    int deg_q;

    /// Target function to be approximated
    Func func;

    /// Interval to be fitted
    Float start, end;

    /// Brake movement (0: diabled - 1: frozen)
    float brake;

    /// Skew initial control points to left or right side
    float skew;

    /// Optimize relative or absolute error?
    bool relerr;

    /// Polynomial coeffs and error const. (deg_p + deg_q + 3 entries)
    std::unique_ptr<Float[]> coeffs;

    /// Roots of the rational polynomial
    std::unique_ptr<Float[]> zeros;

    /// Control points of the current iterate
    std::unique_ptr<Float[]> control;

    /// Temporary storage for LU factorizatoin
    std::unique_ptr<Float[]> A, b;
    std::unique_ptr<int[]> ipiv;

    /// Print debug output while running?
    bool debug;

    Remez(int deg_p, int deg_q, Func func, Float start, Float end,
          float brake, float skew, bool relerr = true, bool debug = false)
        : deg_p(deg_p), deg_q(deg_q), func(func), start(start), end(end),
          brake(brake), skew(skew), relerr(relerr), debug(debug) {
        size_t df = deg_p + deg_q + 2;
        coeffs  = std::unique_ptr<Float[]>(new Float[df + 1]);
        zeros   = std::unique_ptr<Float[]>(new Float[df]);
        control = std::unique_ptr<Float[]>(new Float[df]);
        A       = std::unique_ptr<Float[]>(new Float[df*df]);
        b       = std::unique_ptr<Float[]>(new Float[df]);
        ipiv    = std::unique_ptr<int[]>(new int[df]);
    }

    /**
     * Fit the function 'func' using a rational polynomial so that it exactly
     * interpolates 'func' at the roots of a Chebyshev polynomial with suitable
     * degree.
     */
    bool init() {
        int df = deg_p + deg_q + 1;

        for (int i = 0; i < df; ++i) {
#if 1
            Float xi = cos_(M_PI * (2 * (df - i) - 1) / Float(2 * df));
#else
            Float xi = i / Float(df - 1) * 2 - 1;
#endif
            if (skew != 0) {
                if (skew > 0)
                    xi = -xi;
                xi = pow_((xi + 1) * 0.5, abs_(skew)) * 2 - 1;
                if (skew > 0)
                    xi = -xi;
            }

            Float xi_off = 0.5 * (xi * (end - start) + (start + end)),
                  fxi    = func(xi_off);

            b[i] = fxi;
            control[i] = zeros[i] = xi;

            Float value = 1;
            for (int j = 0; j <= deg_p; ++j) {
                A[i * df + j] = value;
                value *= xi;
            }

            value = xi;
            for (int j = deg_p + 1; j < df; ++j) {
                A[i * df + j] = -value * fxi;
                value *= xi;
            }
        }

        if (!lu(df, A.get(), ipiv.get())) {
            fprintf(stderr, "Remez::init(): lu() failed!\n");
            return false;
        }

        lu_solve(df, A.get(), b.get(), coeffs.get(), ipiv.get());

        for (int i = df; i != deg_p + 1; --i)
            coeffs[i] = coeffs[i - 1];
        coeffs[deg_p + 1] = 1;

        /* Unused */
        coeffs[df + 1] = 0;
        control[df] = zeros[df] = control[df - 1];

        return true;
    }

    Float error(Float x) {
        Float ref = func(0.5 * (x * (end - start) + (start + end))),
              num = horner(x, coeffs.get(), deg_p),
              denom = horner(x, coeffs.get() + deg_p + 1, deg_q);
        Float err = num / denom - ref;
        if (relerr)
            err /= ref;
        return err;
    }

    Float error() const { return abs_(coeffs[deg_p + deg_q + 2]); }

    bool find_control_points() {
        control[0] = -1;
        control[deg_p + deg_q + 1] = 1;

        for (int i = 0; i < deg_p + deg_q; ++i) {
            Float x = golden(zeros[i], zeros[i + 1],
                              [&](Float z) { return -abs_(error(z)); });
            control[i + 1] = control[i + 1] * brake + x * (1 - brake);
        }
        return true;
    }

    bool find_zeros() {
        for (int i = 0; i < deg_p + deg_q + 1; ++i) {
            zeros[i] = illinois(control[i], control[i + 1],
                                [&](Float z) { return error(z); });
            if (isnan_(zeros[i])) {
                fprintf(stderr, "Remez::find_zeros(): failed!\n");
                return false;
            }
        }
        return true;
    }

    bool remez(Float err_guess = 0) {
        int df = deg_p + deg_q + 2;
        Float sign = -1;

        for (int i = 0; i < df; ++i) {
            Float xi  = control[i],
                  xi_off = 0.5 * (xi * (end - start) + (start + end)),
                  fxi = func(xi_off),
                  E = sign;

            if (relerr)
                E *= abs_(fxi);

            b[i] = fxi;

            Float value = 1.0;
            for (int j = 0; j <= deg_p; ++j) {
                A[i * df + j] = value;
                value *= xi;
            }

            value = xi;
            for (int j = deg_p + 1; j <= deg_p + deg_q; ++j) {
                A[i * df + j] = value * (-fxi + E * err_guess);
                value *= xi;
            }

            A[i * df + deg_p + deg_q + 1] = E;
            sign *= -1;
        }

        if (!lu(df, A.get(), ipiv.get())) {
            fprintf(stderr, "Remez::remez(): lu() failed!\n");
            return false;
        }

        lu_solve(df, A.get(), b.get(), coeffs.get(), ipiv.get());

        for (int i = df; i != deg_p + 1; --i)
            coeffs[i] = coeffs[i - 1];
        coeffs[deg_p + 1] = 1;

        return true;
    }

    bool remez_adaptive() {
        Float err_guess = 0;
        int it = 0;

        while (true) {
            if (!remez(err_guess)) {
                fprintf(stderr, "Remez::remez_adaptive(): remez() failed!\n");
                return false;
            }

            if (deg_q == 0)
                break;

            Float err_guess_new   = coeffs[deg_p + deg_q + 2],
                  a_err_guess     = abs_(err_guess),
                  a_err_guess_new = abs_(err_guess_new),
                  err_diff        = abs_(a_err_guess - a_err_guess_new),
                  err_min         = minimum_(a_err_guess, a_err_guess_new);

            if (debug)
                mpfr_printf("    (* error=%.5Re -> %.5Re *)\n", a_err_guess.value,
                            a_err_guess_new.value);

            if (err_diff < 1e-5 * err_min)
                break;

            if (++it == 100) {
                fprintf(stderr, "Remez::remez_adaptive(): warning -- iteration "
                                "count limit reached!\n");
                break;
            }

            err_guess = err_guess_new;
        }

        return true;
    }

    void dump() {
        auto result = domain_shift();
        printf("PlotRemez[{");
        for (int i = 0; i <= deg_p; ++i) {
            result[i].print_mathematica();
            if (i < deg_p)
                fputs(", ", stdout);
        }
        printf("}, {");
        for (int i = 0; i <= deg_q; ++i) {
            result[i + deg_p + 1].print_mathematica();
            if (i < deg_q)
                fputs(", ", stdout);
        }
        printf("}, {");
        for (int i = 0; i < deg_p + deg_q + 2; ++i) {
            Float value = (control[i] * (end - start) + (end + start)) / 2;
            value.print_mathematica();
            if (i < deg_p + deg_q + 1)
                fputs(", ", stdout);
        }
        fputs("}, ", stdout);
        start.print_mathematica();
        fputs(", ", stdout);
        end.print_mathematica();
        fprintf(stdout, ", %i, %i]\n", relerr ? 1 : 0,
                2*((int) ceil(mpfr_get_default_prec() * log(2) / log(10)) + 1));
    }

    bool run() {
        if (!init())
            return false;
        if (debug) {
            printf("(* Initialization with roots *)\n");
            dump();
        }
        if (!find_control_points())
            return false;
        if (debug) {
            printf("(* Initialization with control points *)\n");
            dump();
        }

        Float err_guess = 0;
        int it = 1;
        while (true) {
            if (!remez_adaptive())
                return false;

            if (debug) {
                printf("(* Remez iteration %i *)\n", it);
                dump();
            }

            Float err_guess_new = coeffs[deg_p + deg_q + 2],
                  a_err_guess     = abs_(err_guess),
                  a_err_guess_new = abs_(err_guess_new),
                  err_diff        = abs_(a_err_guess - a_err_guess_new),
                  err_min         = minimum_(a_err_guess, a_err_guess_new);

            if (debug)
                mpfr_printf("(* error=%.5Re -> %.5Re *)\n", a_err_guess.value,
                            a_err_guess_new.value);

            err_guess = a_err_guess_new;

            if (err_diff < 1e-5 * err_min)
                break;

            err_guess = a_err_guess_new;

            if (it == 100 || stop) {
                fprintf(stderr, "Remez::run(): warning -- iteration count "
                                "limit reached!\n");
                return false;
            }

            if (!find_zeros())
                return false;
            if (!find_control_points())
                return false;
            it++;
        }

        return true;
    }

    std::unique_ptr<Float[]> domain_shift() const {
        std::unique_ptr<Float[]> result(new Float[deg_p + deg_q + 3]);
        result[deg_p + deg_q + 2] = coeffs[deg_p + deg_q + 2];

        if (start == -1 && end == 1) {
            for (int i = 0; i < deg_p + deg_q + 2; ++i)
                result[i] = coeffs[i];
            return result;
        } else {
            for (int i = 0; i < deg_p + deg_q + 2; ++i)
                result[i] = 0;
        }

        Float a = 2 / (end - start),
              b = (end + start) / (start - end);

        for (int k = 0; k < 2; ++k) {
            int n = k == 0 ? (deg_p + 1) : (deg_q + 1);
            int shift = k == 0 ? 0 : (deg_p + 1);
            Float *src = coeffs.get() + shift,
                  *dst = result.get() + shift;

            for (int i = 0; i < n; ++i) {
                Float weight = src[i], binom = 1;

                for (int j = 0; j <= i; ++j) {
                    dst[j] += pow_(a, Float(j)) *
                              pow_(b, Float(i - j)) * weight * binom;

                    binom = (binom * (i - j)) / (j + 1);
                }
            }
        }

        for (int i = 0; i < deg_p + deg_q + 2; ++i) {
            if (i != deg_p + 1)
                result[i] /= result[deg_p + 1];
        }
        result[deg_p + 1] = 1;

        return result;
    }

    void apply_domain_shift() {
        auto shifted = domain_shift();
        coeffs.swap(shifted);
    }

    template <typename Target>
    Annealer<Func, Target> anneal(size_t sample_count, size_t iterations,
                                  size_t cycles, bool estrin) {
        return Annealer<Func, Target>(deg_p, deg_q, func, start, end,
                                      coeffs.get(), sample_count, iterations,
                                      cycles, estrin);
    }
};
