#include "remez.h"

/* Declare the function to be optimized here. Underscored math functions
   (defined in remez.h) work on all precisions. Note that Float is
   typically an alias for quadruple precision. */

#if 0
Float func_a = 0, func_b = 1; /// Target interval

Float func(Float x) {
    if (x == 0)
        return 2 / sqrt_(Pi);

    x = sqrt_(x);
    return erf_(x) / x;
};
#else
Float func_a = 1, func_b = 6; /// Target interval (6 for double, 4 for single prec.)

Float func(Float x) {
    return log2_(erfc_(x)) / x;
};
#endif


int main(int argc, char **argv) {
    bool anneal = false,
         estrin = true,
         double_precision = false,
         verbose = false,
         relerr = true,
         check = false,
         help = false;

    int deg_p_min = 0, deg_p_max = -1,
        deg_q_min = 0, deg_q_max = -1,
        deg_max = 30, def_prec = 1024;

    size_t anneal_samples    = 100000,
           anneal_iterations = 100000;

    float brake = 0, skew = 1;

    int c;
    char *endptr = nullptr;
    char *check_poly = nullptr;

    while ((c = getopt(argc, argv, "dehHvAcm:p:q:as:i:C:P:b:S:")) != -1) {
        switch (c) {
            case 'd': double_precision = true; break;
            case 'H': estrin = false; break;
            case 'h': help = true; break;
            case 'v': verbose = true; break;
            case 'A': relerr = false; break;
            case 'c':
                check = true;
                break;
            case 'C':
                check_poly = strdup(optarg);
                break;
            case 'm':
                deg_max = strtod(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "Could not parse integer!\n");
                    return -1;
                }
                break;
            case 'p':
                deg_p_min = deg_p_max = strtod(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "Could not parse integer!\n");
                    return -1;
                }
                break;
            case 'b':
                brake = strtof(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "Could not parse floating point value !\n");
                    return -1;
                }
                break;
            case 'S':
                skew = strtof(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "Could not parse floating point value !\n");
                    return -1;
                }
                break;
            case 'q':
                deg_q_min = deg_q_max = strtod(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "Could not parse integer!\n");
                    return -1;
                }
                break;
            case 'P':
                def_prec = strtod(optarg, &endptr);
                if (endptr == optarg) {
                    fprintf(stderr, "Could not parse integer!\n");
                    return -1;
                }
                break;
            case 'a':
                anneal = true;
                break;
            case 's':
                anneal_samples = (size_t) strtoull(optarg, &endptr, 10);
                if (endptr == optarg) {
                    fprintf(stderr, "Could not parse integer!\n");
                    return -1;
                }
                break;
            case 'i':
                anneal_iterations = (size_t) strtoull(optarg, &endptr, 10);
                if (endptr == optarg) {
                    fprintf(stderr, "Could not parse integer!\n");
                    return -1;
                }
                break;
            default:
                help = true;
                break;

        }
    }

    if (help) {
        printf("Syntax: %s [options]\n\n", argv[0]);
        printf("Generates a polynomial or rational polynomial fit using the Remez algorithm.\n");
        printf("The resulting precision polynomial is then annealed to find a rounded variant\n");
        printf("that performs well for a given target precision and evaluation scheme. By\n");
        printf("default, the implementation tries many different configurations and prints the\n");
        printf("theoretical and actual achieved error in max/mean ULPs. After selecting the\n");
        printf("desired configuration, re-run the program with '-a' to start the simulated\n");
        printf("annealing procedure.\n\n");

        printf("Search-related parameters:\n\n");
        printf("   -v          Generate verbose output.\n\n");
        printf("   -p value    Fix the degree of the numerator (must be >= 0).\n\n");
        printf("   -q value    Fix the degree of the denominator (must be >= 0).\n\n");
        printf("   -m value    Set the maximum degree used in the initial search.\n\n");
        printf("   -P value    Mantissa bits used for optimization (default: 1024).\n\n");
        printf("   -A          Optimize absolute instead of relative error.\n\n");
        printf("Tricks to improve convergence (esp. for rational polynomials):\n\n");
        printf("   -S value    Skew the initial point set to the left (< 1) or right (> 1).\n\n");
        printf("   -b value    Limit control point motion ([0, 1], where 0=off and 1=frozen).\n\n");
        printf("Annealing-related parameters:\n\n");
        printf("   -H        Target Horner evalution instead of Estrin's parallel scheme.\n\n");
        printf("   -d        Round to double precision (default: single precision).\n\n");
        printf("   -s value  Number of sample evaluations (default: 100000)\n\n");
        printf("   -i value  Number of annealing iterations (default: 100000)\n\n");
        printf("   -c        After annealing, do an exhaustive accuracy check.\n\n");
        printf("   -C a,b,.. Check the accuracy of the provided polynomial.\n");
        printf("             (can be combined with -c).\n\n");
        return -1;
    }

    if (deg_p_max == -1)
        deg_p_max = deg_max;
    if (deg_q_max == -1)
        deg_q_max = deg_max;

    mpfr_set_default_prec(def_prec);

    printf("-----------------------------------------------\n");
    printf("deg(P)               = %i .. %i\n", deg_p_min, deg_p_max);
    printf("deg(Q)               = %i .. %i\n", deg_q_min, deg_q_max);
    printf("precision            = %i mantissa bits\n", def_prec);
    printf("target precision     = %s precision\n", double_precision ? "double" : "single");
    printf("relative error       = %s\n", relerr ? "yes" : "no");
    printf("evaluation scheme    = %s\n", estrin ? "estrin" : "horner");
    printf("brake control points = %f\n", brake);
    printf("anneal               = %s\n", anneal ? "yes" : "no");
    printf("annealing iterations = %zu\n", anneal_iterations);
    printf("annealing samples    = %zu\n", anneal_samples);
    printf("brute force check    = %s\n", check ? "yes" : "no");
    printf("-----------------------------------------------\n");

    if (check && double_precision) {
        fprintf(stderr,
                "Error: an exhaustive error check is only feasible in single precision!\n");
        return -1;
    }

    if (sizeof(Float) == 4 && double_precision) {
        fprintf(stderr,
                "Error: double precision optimization requires that this "
                "program is compiled using quadruple precision.\n");
        return -1;
    }

    if (verbose) {
        printf(R"m(PlotRemez[num_, den_, control_, start_, end_, relerr_, prec_] :=
 Block[{p, q, e, err, cp, p1, p2},
  p[x_] = HornerForm[Sum[num[[i]]*x^(i - 1), {i, 1, Length[num]}], x];
  q[x_] = HornerForm[Sum[den[[i]]*x^(i - 1), {i, 1, Length[den]}], x];
  e[x_] := (p[x]/q[x] - Ref[x])*If[relerr == 1, 1/Ref[x], 1];
  err = SetPrecision[
    Max[Table[Abs[e[x]], {x, start, end, (end - start)/100}]], 38];
  cp = Table[{PointSize[0.02],
     Point[{control[[k]], e[SetPrecision[control[[k]], prec]]/err}]}, {k, 1,
     Length[control]}];
  p1 = Plot[e[SetPrecision[x, prec]] / err, {x, start, end}, PlotRange -> Full,
   Epilog -> cp, PlotLegends -> Placed[LineLegend[{"Error \[Cross] ".  N[err, 4]}], Bottom],
   Frame -> True, ImageSize -> 400];
  p2 = Plot[{p[x], q[x]}, {x, start, end}, PlotRange -> Full, Frame -> True,
   PlotLegends -> Placed[LineLegend[{"p(x)", "q(x)"}], Bottom], ImageSize -> 400];
  GraphicsRow[{p1, p2}, Spacings -> {50, 0}, ImageSize -> {850, 400}]
]
)m");
    }

    if (check_poly) {
        if (deg_q_min != deg_q_max ||
            deg_p_min != deg_p_max) {
            fprintf(stderr, "-C: must also specify degrees via -p and -q!\n");
            return -1;
        }

        Remez<decltype(&func)> remez(deg_p_min, deg_q_min, func, func_a, func_b,
                                     brake, skew, relerr, verbose);

        char *saveptr = nullptr;
        char *token = strtok_r(check_poly, ", ", &saveptr);
        int tokens = 0;
        while (token) {
            if (tokens == deg_p_min + deg_q_min + 2) {
                fprintf(stderr,
                        "The provided polynomial has too many coefficients!\n");
                return -1;
            }

            remez.coeffs[tokens++] = Float(strtod(token, &endptr));
            if (endptr == optarg) {
                fprintf(stderr, "Could not parse floating point value!\n");
                return -1;
            }
            token = strtok_r(nullptr, ", ", &saveptr);
        }

        if (deg_q_min > 0) {
            if (tokens != deg_p_min + deg_q_min + 2) {
                fprintf(stderr,
                        "The provided polynomial has too few coefficients!\n");
                return -1;
            }
        } else {
            if (tokens != deg_p_min + 1) {
                fprintf(stderr,
                        "The provided polynomial has too few coefficients!\n");
                return -1;
            }

            remez.coeffs[deg_p_min + 1] = 1;
        }

        if (double_precision) {
            auto annealer =
                remez.anneal<double>(anneal_samples, anneal_iterations, estrin);

            annealer.dump();

            if (anneal) {
                annealer.go();
                annealer.dump();
            }

            if (check) {
                annealer.check();
            } else {
                printf("max=%.3f ulp, avg=%.3f ulp.\n", annealer.err_best.first,
                       annealer.err_best.second);
            }
        } else {
            auto annealer =
                remez.anneal<float>(anneal_samples, anneal_iterations, estrin);

            annealer.dump();

            if (anneal) {
                annealer.go();
                annealer.dump();
            }

            if (check) {
                annealer.check();
            } else {
                printf("max=%.3f ulp, avg=%.3f ulp.\n", annealer.err_best.first,
                       annealer.err_best.second);
            }
        }

        return 0;
    }

    signal(SIGINT, [](int){ stop = true; });

    for (int q = deg_q_min; q <= deg_q_max && !stop; ++q) {
        for (int p = deg_p_min; p <= deg_p_max && !stop; ++p) {
            Remez<decltype(&func)> remez(p, q, func, func_a, func_b, brake,
                                         skew, relerr, verbose);
            if (!remez.run())
                continue;

            remez.apply_domain_shift();

            mpfr_printf("P=%i, Q=%i: %.3Re -> ", p, q, remez.error().value);

            if (double_precision) {
                auto annealer = remez.anneal<double>(anneal_samples,
                                                     anneal_iterations, estrin);
                if (anneal) {
                    annealer.go();
                    annealer.dump();
                }

                if (check) {
                    annealer.check();
                } else {
                    printf("max=%.3f ulp, avg=%.3f ulp.\n", annealer.err_best.first,
                           annealer.err_best.second);
                }
            } else {
                auto annealer = remez.anneal<float>(anneal_samples,
                                                    anneal_iterations, estrin);
                if (anneal) {
                    annealer.go();
                    annealer.dump();
                }

                if (check) {
                    annealer.check();
                } else {
                    printf("max=%.3f ulp, avg=%.3f ulp.\n", annealer.err_best.first,
                           annealer.err_best.second);
                }
            }
        }
    }
}
