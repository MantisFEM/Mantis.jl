using Test

const COLOR_GREEN = "\x1b[32m"
const COLOR_RED = "\x1b[31m"
const COLOR_RESET = "\x1b[0m"

macro expected(expected_dict, test_func, match_op=isequal)
    return esc(
        quote
            all_passed = true
            for k in keys($expected_dict)
                val = $test_func(k)
                exp_val = $expected_dict[k]
                if !($match_op(val, exp_val))
                    println(
                        stderr,
                        "$(COLOR_RED)Error on key $(k)$(COLOR_RESET).\n",
                        "Expected: ",
                        "$(COLOR_GREEN)$(exp_val)$(COLOR_RESET)\n",
                        "Obtained: ",
                        "$(COLOR_RED)$(val)$(COLOR_RESET)",
                    )
                    all_passed = false
                end
            end

            all_passed
        end,
    )
end

export expected

