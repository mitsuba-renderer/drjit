#include "remez.h"

/* Declare the function to be optimized here. Underscored math functions
   (defined in remez.h) work on all precisions. Note that Float is
   typically an alias for quadruple precision. */
Float func(Float x) {
    if (x == 0)
        return 2 / sqrt_(Pi);

    x = sqrt_(x);
    return erf_(x) / x;
};

/// Left interval endpoint of the approximation
Float func_a = 0;

/// Right interval endpoint of the approximation
Float func_b = 1;


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
        deg_max = 10;

    size_t anneal_samples    = 100000,
           anneal_iterations = 100000;

    int c;
    char *endptr = nullptr;
    while ((c = getopt(argc, argv, "dehHvAcm:p:q:as:i:")) != -1) {
        switch (c) {
            case 'd': double_precision = true; break;
            case 'H': estrin = false; break;
            case 'h': help = true; break;
            case 'v': verbose = true; break;
            case 'A': relerr = false; break;
            case 'c': check = true; break;
            case 'm': deg_max = strtod(optarg, &endptr);
                      if (endptr == optarg) {
                          fprintf(stderr, "Could not parse integer!\n");
                          return -1;
                      }
                      break;
            case 'p': deg_p_min = deg_p_max = strtod(optarg, &endptr);
                      if (endptr == optarg) {
                          fprintf(stderr, "Could not parse integer!\n");
                          return -1;
                      }
                      break;
            case 'q': deg_q_min = deg_q_max = strtod(optarg, &endptr);
                      if (endptr == optarg) {
                          fprintf(stderr, "Could not parse integer!\n");
                          return -1;
                      }
                      break;
            case 'a': anneal = true; break;
            case 's': anneal_samples = (size_t) strtoull(optarg, &endptr, 10);
                      if (endptr == optarg) {
                          fprintf(stderr, "Could not parse integer!\n");
                          return -1;
                      }
                      break;
            case 'i': anneal_iterations = (size_t) strtoull(optarg, &endptr, 10);
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
        printf("   -A          Optimize absolute instead of relative error.\n\n");
        printf("   -p value    Fix the degree of the numerator (must be >= 0).\n\n");
        printf("   -q value    Fix the degree of the denominator (must be >= 0).\n\n");
        printf("   -m value    Set the maximum degree used in the initial search.\n\n");
        printf("Annealing-related parameters:\n\n");
        printf("   -H        Target Horner evalution instead of Estrin's parallel scheme.\n\n");
        printf("   -d        Round to double precision (default: single precision).\n\n");
        printf("   -s value  Number of sample evaluations (default: 100000)\n\n");
        printf("   -i value  Number of annealing iterations (default: 100000)\n\n");
        printf("   -c        After annealing, do an exhaustive accuracy check.\n\n");
        return -1;
    }

    if (deg_p_max == -1)
        deg_p_max = deg_max;
    if (deg_q_max == -1)
        deg_q_max = deg_max;

    printf("-----------------------------------------------\n");
    printf("deg(P)               = %i .. %i\n", deg_p_min, deg_p_max);
    printf("deg(Q)               = %i .. %i\n", deg_q_min, deg_q_max);
    printf("precision            = %s\n", double_precision ? "double" : "single");
    printf("relative error       = %s\n", relerr ? "yes" : "no");
    printf("evaluation scheme    = %s\n", estrin ? "estrin" : "horner");
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
        printf("(* Pipe the output of this program through remez-filter.py\n"
               "to convert exponents to Mathematica notation *)\n\n");
        printf(R"m(PlotRemez[num_, den_, control_, start_, end_, relerr_] :=
 Block[{n, d, e, err, cp},
  n[x_] = HornerForm[Sum[num[[i]]*x^(i - 1), {i, 1, Length[num]}], x];
  d[x_] = HornerForm[Sum[den[[i]]*x^(i - 1), {i, 1, Length[den]}], x];
  e[x_] := (n[x]/d[x] - Ref[x])*If[relerr == 1, 1/Ref[x], 1];
  err = SetPrecision[
    Max[Table[Abs[e[x]], {x, start, end, (end - start)/100}]], 38];
  cp = Table[{PointSize[0.02],
     Point[{control[[k]], e[control[[k]]]/err}]}, {k, 1,
     Length[control]}];
  Plot[e[x]/err, {x, start, end}, PlotRange -> Full, Epilog -> cp,
   WorkingPrecision -> 38,
   FrameLabel -> Style["\[Cross]".N[err, 3], Medium], Frame -> True]
]
)m");
    }

    signal(SIGINT, [](int){ stop = true; });

    for (int q = deg_q_min; q <= deg_q_max && !stop; ++q) {
        for (int p = deg_p_min; p <= deg_p_max && !stop; ++p) {
            Remez<decltype(&func)> remez(p, q, func, func_a, func_b, relerr,
                                         verbose);
            if (!remez.run())
                continue;

            printf("P=%i, Q=%i: " FLT_SHORT " -> ", p, q, remez.error());

            if (double_precision) {
                auto annealer = remez.anneal<double>(anneal_samples,
                                                     anneal_iterations, estrin);
                if (anneal) {
                    annealer.go();
                    annealer.dump();
                }

                printf("max=%f ulp, avg=%f ulp.\n", annealer.err_best.first,
                       annealer.err_best.second);

                if (check)
                    annealer.check();
            } else {
                auto annealer = remez.anneal<float>(anneal_samples,
                                                    anneal_iterations, estrin);
                if (anneal) {
                    annealer.go();
                    annealer.dump();

                }

                printf("max=%.3f ulp, avg=%.3f ulp.\n", annealer.err_best.first,
                       annealer.err_best.second);

                if (check)
                    annealer.check();
            }
        }
    }
}
