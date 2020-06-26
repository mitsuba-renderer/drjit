#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <atomic>
#include <random>
#include <signal.h>
#include <unistd.h>

// Perform all arithmetic using quadruple precision
#define QUAD_PREC 1

#if !defined(__clang__) && QUAD_PREC == 0
#include <quadmath.h>

#pragma GCC diagnostic ignored "-Wformat"
#pragma GCC diagnostic ignored "-Wformat-extra-args"

using Float = __float128;
#define FLT "%.38Qg`38"
#define FLT_SHORT "%.4Qe"
#define Pi M_PIq
inline Float fma_(Float a, Float b, Float c) { return fmaq(a, b, c); }
inline Float log_(Float a) { return logq(a); }
inline Float abs_(Float a) { return fabsq(a); }
inline Float sqrt_(Float a) { return sqrtq(a); }
inline Float exp_(Float a) { return expq(a); }
inline Float erf_(Float a) { return erfq(a); }
inline Float sin_(Float a) { return sinq(a); }
inline Float cos_(Float a) { return cosq(a); }
inline Float min_(Float a, Float b) { return fminq(a, b); }
inline Float max_(Float a, Float b) { return fmaxq(a, b); }
inline bool isnan_(Float a) { return isnanq(a); }
#else
using Float = double;
#define FLT "%.17g"
#define FLT_SHORT "%.4e"
#define Pi M_PI
#endif

static bool stop = false;

inline float fma_(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
inline float log_(float a) { return logf(a); }
inline float abs_(float a) { return fabsf(a); }
inline float sqrt_(float a) { return sqrtf(a); }
inline float exp_(float a) { return expf(a); }
inline float sin_(float a) { return sinf(a); }
inline float cos_(float a) { return cosf(a); }
inline bool isnan_(float a) { return isnanf(a); }
inline float min_(float a, float b) { return __builtin_fminf(a, b); }
inline float max_(float a, float b) { return __builtin_fmaxf(a, b); }

inline double fma_(double a, double b, double c) { return __builtin_fma(a, b, c); }
inline double log_(double a) { return log(a); }
inline double abs_(double a) { return fabs(a); }
inline double sqrt_(double a) { return sqrt(a); }
inline double exp_(double a) { return exp(a); }
inline double erf_(double a) { return erf(a); }
inline double sin_(double a) { return sin(a); }
inline double cos_(double a) { return cos(a); }
inline bool isnan_(double a) { return std::isnan(a); }
inline double min_(double a, double b) { return __builtin_fmin(a, b); }
inline double max_(double a, double b) { return __builtin_fmax(a, b); }


// Horner-style evaluation of a polynomial of degree 'n'
template <typename Value> Value horner(Value x, Value *coeffs, int n) {
    Value accum = coeffs[n];
    for (int i = 1; i <= n; ++i)
        accum = fma_(x, accum, coeffs[n - i]);
    return accum;
}

// Estrin-style evaluation of a polynomial of degree 'n'
template <typename Value> Value estrin(Value x, Value *coeffs, int n) {
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


template <typename Func>
Float golden(Float a, Float b, const Func &f) {
    const Float invphi        = 0.6180339887498948,
                invphi2       = 0.3819660112501052,
                invlog_invphi = -2.078086921235028,
                tol           = 1e-5;

    Float h = b - a;
    int n_steps = (int) log_(tol / h) * invlog_invphi;

    Float c = a + invphi2 * h,
           d = a + invphi * h,
           fc = f(c),
           fd = f(d);

    for (int i = 0; i < n_steps; ++i) {
        if (fc < fd) {
            b = d; d = c; fd = fc;
            h = invphi * h;
            c = a + invphi2 * h;
            fc = f(c);
        } else {
            a = c; c = d; fc = fd;
            h = invphi * h;
            d = a + invphi * h;
            fd = f(d);
        }
    }

    return .5 * ((fc < fd) ? (a + d) : (c + b));
}

// Variant on the false position method, finds a root on [a, b]
template <typename Func> Float illinois(Float a, Float b, const Func &f) {
    Float fa = f(a), fb = f(b);

    if (!(fa * fb < 0)) {
        fprintf(stderr,
                "illinos(" FLT_SHORT ", " FLT_SHORT "): called with a non-bracketing "
                "interval (" FLT_SHORT ", " FLT_SHORT ")!\n", a, b, fa, fb);
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
    float temp_initial;
    float temp_end;
    bool estrin;

    /// Reference values
    std::unique_ptr<Target[]> x;
    std::unique_ptr<Float[]> x_ulp;
    std::unique_ptr<Float[]> y;

    Annealer(int deg_p, int deg_q, Func func, Float start, Float end,
             Float *coeffs_, size_t sample_count, size_t iterations,
             bool estrin)
        : deg_p(deg_p), deg_q(deg_q), func(func),
          start((Target) start), end((Target) end),
          coeffs_cur(new Target[deg_p + deg_q + 2]),
          coeffs_prop(new Target[deg_p + deg_q + 2]),
          coeffs_best(new Target[deg_p + deg_q + 2]),
          sample_count(sample_count),
          iterations(iterations),
          estrin(estrin),
          x(new Target[sample_count]),
          x_ulp(new Float[sample_count]),
          y(new Float[sample_count]) {

        fflush(stdout);
        for (int i = 0; i < deg_p + deg_q + 2; ++i)
            coeffs_cur[i] = (Target) coeffs_[i];
        coeffs_prop[deg_p + 1] = 1.0;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < sample_count; ++i) {
            Float xf = start + (end - start) * Float(i) / Float(sample_count - 1);
            Target xt = Target(xf);
            Float yf = func(Float(xt));
            Target yt = Target(yf);

            x[i] = xt;
            y[i] = yf;
            x_ulp[i] = (Float) std::nextafter(yt, std::numeric_limits<Target>::infinity()) - yt;
        }
        err_cur = err_best = error(coeffs_cur.get());
        memcpy(coeffs_best.get(), coeffs_cur.get(),
               sizeof(Target) * (deg_p + deg_q + 2));
    }

    void dump() {
        const char *fmt = sizeof(float) == 4 ? "        %s%a // %s%.9e%s\n"
                                             : "        %s%a // %s%.17e%s\n";
        printf("\n   P = %s(x,\n", estrin ? "estrin" : "horner");
        for (int i = 0; i <= deg_p; ++i)
            printf(fmt,
                   coeffs_best[i] >= 0 ? " " : "", coeffs_best[i],
                   coeffs_best[i] >= 0 ? " " : "", coeffs_best[i],
                   i < deg_p ? "," : "");
        printf("   )\n   Q = %s(x,\n", estrin ? "estrin" : "horner");
        for (int i = 0; i <= deg_q; ++i)
            printf(fmt,
                   coeffs_best[i + deg_p + 1] >= 0 ? " " : "", coeffs_best[i + deg_p + 1],
                   coeffs_best[i + deg_p + 1] >= 0 ? " " : "", coeffs_best[i + deg_p + 1],
                   i < deg_q ? "," : "");
        printf("   )\n");
    }

    std::pair<float, float> error(Target *c) const {
        double err_sum = 0;
        AtomicFloat<double> err_max(0);

        if (estrin) {
            #pragma omp parallel for schedule(static) reduction(+:err_sum)
            for (size_t i = 0; i < sample_count; ++i) {
                Target xt    = x[i],
                       num   = ::estrin(xt, c , deg_p),
                       denom = ::estrin(xt, c + deg_p + 1, deg_q),
                       value = num / denom;

                double err = double(abs_((Float) value - y[i]) / x_ulp[i]);

                err_max.max(err);
                err_sum += err;
            }
        } else {
            #pragma omp parallel for schedule(static) reduction(+:err_sum)
            for (size_t i = 0; i < sample_count; ++i) {
                Target xt    = x[i],
                       num   = ::horner(xt, c , deg_p),
                       denom = ::horner(xt, c + deg_p + 1, deg_q),
                       value = num / denom;

                double err = double(abs_((Float) value - y[i]) / x_ulp[i]);

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
                       num   = ::estrin(xt, coeffs_best.get() , deg_p),
                       denom = ::estrin(xt, coeffs_best.get() + deg_p + 1, deg_q),
                       value = num / denom,
                       ulp   = std::nextafter(value, std::numeric_limits<Target>::infinity()) - value;

                double err = double(abs_((Float) value - func((Float) xt)) / ulp);

                err_max.max(err);
                err_sum += err;
            }
        } else {
            #pragma omp parallel for schedule(static) reduction(+:err_sum)
            for (Int i = start_i; i < end_i; ++i) {
                if (stop)
                    continue;

                Target xt    = memcpy_cast<Target>(i),
                       num   = ::horner(xt, coeffs_best.get() , deg_p),
                       denom = ::horner(xt, coeffs_best.get() + deg_p + 1, deg_q),
                       value = num / denom,
                       ulp   = std::nextafter(value, std::numeric_limits<Target>::infinity()) - value;

                double err = double(abs_((Float) value - func((Float) xt)) / ulp);

                err_max.max(err);
                err_sum += err;
            }
        }

        printf(" -> exhaustive search yields: max=%.2f ulp, avg=%.3f ulp.\n",
               (double) err_max, err_sum / (end_i - start_i));
    }

    void go() {
        std::mt19937 engine;
        std::uniform_real_distribution<float> uniform;
        std::normal_distribution<float> normal;

        size_t n_invalid = 0, n_accepted = 0, n_rejected = 0;
        float scale = 2;

        printf("\n");
        for (size_t i = 0; i < iterations && !stop; ++i) {
            float temp = 10 * expf(i / -float(iterations - 1) * 5);

            if (i % 1000 == 0)
                printf("   %05zu: max=%.2f ulp, avg=%.3f ulp, temp=%.1f.\n", i,
                       err_cur.first, err_cur.second, temp);

            bool changed = false;
            for (int i = 0; i < deg_p + deg_q + 2; ++i) {
                if (i == deg_p + 1)
                    continue;

                Int value = memcpy_cast<Int>(coeffs_cur[i]);
                int shift = (int) (normal(engine) * temp);
                value += shift;
                if (shift != 0)
                    changed = true;
                coeffs_prop[i] = memcpy_cast<Target>(value);
            }

            if (!changed) {
                n_invalid++;
                memcpy(coeffs_cur.get(), coeffs_best.get(),
                       sizeof(Target) * (deg_p + deg_q + 2));
                err_cur = err_best;
                continue;
            }

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
        printf("   -> %zu invalid, %zu accepted, %zu rejected steps.\n   ",
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
          bool relerr = true, bool debug = false)
        : deg_p(deg_p), deg_q(deg_q), func(func), start(start), end(end),
          relerr(relerr), debug(debug) {
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
            Float node = cos_(M_PI * (2 * (df - i) - 1) / Float(2 * df));
#else
            Float node = i / Float(df - 1) * 2 - 1;
#endif

            Float xi   = 0.5 * (node * (end - start) + (start + end)),
                  fxi  = func(xi);

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
        Float ref = func(x),
              num = horner(x, coeffs.get(), deg_p),
              denom = horner(x, coeffs.get() + deg_p + 1, deg_q);
        Float err = num / denom - ref;
        if (relerr)
            err /= ref;
        return err;
    }

    Float error() const { return abs_(coeffs[deg_p + deg_q + 2]); }

    bool find_control_points() {
        control[0] = start;
        control[deg_p + deg_q + 1] = end;

        for (int i = 0; i < deg_p + deg_q; ++i) {
            Float x = golden(zeros[i], zeros[i + 1],
                              [&](Float z) { return -abs_(error(z)); });
            control[i + 1] = x;
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
                  fxi = func(xi),
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
                  err_min         = min_(a_err_guess, a_err_guess_new);

            // printf("    (* error=" FLT " -> " FLT " *)\n", a_err_guess, a_err_guess_new);

            if (err_diff < 1e-5 * err_min)
                break;

            if (++it == 100) {
                fprintf(stderr, "Remez::remez_adaptive(): warning -- iteration count limit reached!\n");
                break;
            }

            err_guess = err_guess_new;
        }

        return true;
    }

    void dump() {
        printf("PlotRemez[{");
        for (int i = 0; i <= deg_p; ++i)
            printf(FLT "%s", coeffs[i], i < deg_p ? ", " : "");
        printf("}, {");
        for (int i = 0; i <= deg_q; ++i)
            printf(FLT "%s", coeffs[i + deg_p + 1], i < deg_q ? ", " : "");
        printf("}, {");
        for (int i = 0; i < deg_p + deg_q + 2; ++i)
            printf(FLT "%s", control[i], i < deg_p + deg_q + 1 ? ", " : "");
        printf("}, " FLT ", " FLT ", %i]\n", start, end, relerr ? 1 : 0);
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
        for (int i = 0; i< 3; ++i) {
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
                  err_min         = min_(a_err_guess, a_err_guess_new);

            if (err_diff < 1e-5 * err_min)
                break;

            if (++it == 100) {
                fprintf(stderr, "Remez::run(): warning -- iteration count limit reached!\n");
                return false;
            }

            if (!find_zeros())
                return false;
            if (!find_control_points())
                return false;
        }

        return true;
    }

    template <typename Target>
    Annealer<Func, Target> anneal(size_t sample_count, size_t iterations, bool estrin) {
        return Annealer<Func, Target>(deg_p, deg_q, func, start, end,
                                      coeffs.get(), sample_count, iterations, estrin);
    }
};
