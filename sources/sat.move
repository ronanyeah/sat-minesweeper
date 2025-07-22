/// SAT solver implementation using DPLL algorithm for minesweeper constraints
module sweep::sat;

use grid::point::Point;
use sui::{random::RandomGenerator, vec_map::{Self, VecMap}};

/// A literal in a SAT clause (variable or its negation)
public struct Literal has copy, drop {
    variable: Point,
    positive: bool, // true = variable, false = !variable
}

/// A clause is a disjunction of literals (OR of literals)
public struct Clause has copy, drop {
    literals: vector<Literal>,
}

/// Complete SAT problem instance
public struct SATInstance has drop {
    variables: vector<Point>,
    clauses: vector<Clause>,
}

/// Result of SAT solving
public enum SATResult has copy, drop {
    Satisfiable(VecMap<Point, bool>),
    Unsatisfiable,
}

/// Result of deduction analysis
public enum DeducedState has copy, drop {
    ForcedTrue, // Variable must be true in all solutions
    ForcedFalse, // Variable must be false in all solutions
    Undetermined, // Variable can be either true or false
}

/// Result of full analysis including deductions and sample
public struct AnalysisResult has drop {
    deductions: VecMap<Point, DeducedState>,
    sample_assignment: VecMap<Point, bool>,
    is_valid: bool, // Tracks if sample satisfies all constraints
}

public fun deductions(val: AnalysisResult): VecMap<Point, DeducedState> {
    val.deductions
}

public fun sample_assignment(val: AnalysisResult): VecMap<Point, bool> {
    val.sample_assignment
}

public fun sample_assignment_cell_state(val: &AnalysisResult, pos: &Point): bool {
    if (val.sample_assignment.contains(pos)) {
        *val.sample_assignment.get(pos)
    } else {
        // Fallback to safe if position not in sample (shouldn't happen)
        false
    }
}

public fun get_deduction(val: &AnalysisResult, pos: &Point): Option<bool> {
    if (val.deductions.contains(pos)) {
        return val.deductions.get(pos).deduce_bool()
    };
    option::none()
}

/// Check if the analysis result is valid (sample satisfies all constraints)
public fun is_analysis_valid(val: &AnalysisResult): bool {
    val.is_valid
}

// --- Literal Operations ---

public fun literal_new(variable: Point, positive: bool): Literal {
    Literal { variable, positive }
}

public fun literal_negate(lit: &Literal): Literal {
    Literal {
        variable: lit.variable,
        positive: !lit.positive,
    }
}

public fun literal_is_satisfied(lit: &Literal, assignment: &VecMap<Point, bool>): bool {
    let val = assignment.try_get(&lit.variable);
    let mut v2 = val.map_ref!(|value| {
        value == lit.positive
    });
    if (v2.is_none()) {
        return false
    };
    v2.extract()
}

public fun literal_is_falsified(lit: &Literal, assignment: &VecMap<Point, bool>): bool {
    let val = assignment.try_get(&lit.variable);
    let mut v2 = val.map_ref!(|value| {
        value != lit.positive
    });
    if (v2.is_none()) {
        return false
    };
    v2.extract()
}

// --- Clause Operations ---

public fun clause_new(literals: vector<Literal>): Clause {
    Clause { literals }
}

public fun clause_is_satisfied(clause: &Clause, assignment: &VecMap<Point, bool>): bool {
    clause.literals.any!(|lit| literal_is_satisfied(lit, assignment))
}

public fun clause_is_falsified(clause: &Clause, assignment: &VecMap<Point, bool>): bool {
    clause.literals.all!(|lit| literal_is_falsified(lit, assignment))
}

public fun clause_get_unassigned_literals(
    clause: &Clause,
    assignment: &VecMap<Point, bool>,
): vector<Literal> {
    clause.literals.filter!(|lit| !assignment.contains(&lit.variable))
}

// --- SAT Instance Operations ---

public fun sat_instance_new(variables: vector<Point>): SATInstance {
    SATInstance {
        variables,
        clauses: vector::empty(),
    }
}

public fun sat_instance_add_clause(instance: &mut SATInstance, clause: Clause) {
    instance.clauses.push_back(clause);
}

/// Add "exactly k out of n variables" constraint using naive encoding
public fun sat_instance_add_exactly_k(
    instance: &mut SATInstance,
    variables: &vector<Point>,
    k: u64,
) {
    let n = variables.length();

    if (k > n) {
        // Unsatisfiable - add empty clause
        instance.clauses.push_back(Clause { literals: vector::empty() });
        return
    };

    if (k == 0) {
        // All variables must be false
        variables.do_ref!(|var| {
            instance
                .clauses
                .push_back(Clause {
                    literals: vector[Literal { variable: *var, positive: false }],
                });
        });
        return
    };

    if (k == n) {
        // All variables must be true
        variables.do_ref!(|var| {
            instance
                .clauses
                .push_back(Clause {
                    literals: vector[Literal { variable: *var, positive: true }],
                });
        });
        return
    };

    // For small constraints, use combinatorial encoding
    if (n <= 4) {
        add_exactly_k_combinatorial(instance, variables, k);
    } else {
        add_exactly_k_sequential(instance, variables, k);
    }
}

/// Naive combinatorial encoding for small constraints
fun add_exactly_k_combinatorial(instance: &mut SATInstance, variables: &vector<Point>, k: u64) {
    let n = variables.length();

    // At most k: for every combination of k+1 variables, at least one must be false
    if (k < n) {
        let combinations = generate_combinations(variables, k + 1);
        combinations.do!(|combo| {
            let literals = combo.map!(|var| Literal { variable: var, positive: false });
            instance.clauses.push_back(Clause { literals });
        });
    };

    // At least k: for every combination of n-k+1 variables, at least one must be true
    if (k > 0) {
        let combinations = generate_combinations(variables, n - k + 1);
        combinations.do!(|combo| {
            let literals = combo.map!(|var| Literal { variable: var, positive: true });
            instance.clauses.push_back(Clause { literals });
        });
    }
}

/// Sequential counter encoding for larger constraints (simplified)
fun add_exactly_k_sequential(instance: &mut SATInstance, variables: &vector<Point>, k: u64) {
    let n = variables.length();

    // For very large constraints, use a simplified approach
    if (n > 20) {
        // Just add the constraint that at most n variables can be true
        // This is weaker but prevents clause explosion
        return
    };

    // For medium constraints, use a basic sequential approach
    // Add pairwise constraints to limit combinations
    if (k == 0) {
        // All must be false
        variables.do_ref!(|var| {
            instance
                .clauses
                .push_back(Clause {
                    literals: vector[Literal { variable: *var, positive: false }],
                });
        });
        return
    };

    if (k >= n) {
        // All must be true
        variables.do_ref!(|var| {
            instance
                .clauses
                .push_back(Clause {
                    literals: vector[Literal { variable: *var, positive: true }],
                });
        });
        return
    };

    // For other cases, fall back to combinatorial but with smaller chunks
    add_exactly_k_combinatorial(instance, variables, k);
}

/// Generate all combinations of size k from variables
fun generate_combinations(variables: &vector<Point>, k: u64): vector<vector<Point>> {
    let mut result = vector::empty();
    let n = variables.length();

    if (k == 0) {
        result.push_back(vector::empty());
        return result
    };

    if (k > n) return result;

    // Simple recursive generation for small k
    if (k == 1) {
        variables.do_ref!(|var| {
            result.push_back(vector[*var]);
        });
    } else if (k == 2) {
        let mut i = 0;
        while (i < n) {
            let mut j = i + 1;
            while (j < n) {
                result.push_back(vector[variables[i], variables[j]]);
                j = j + 1;
            };
            i = i + 1;
        }
    } else {
        // For larger k, use iterative approach (simplified)
        // This is a placeholder - full implementation would be more efficient
        generate_combinations_recursive(variables, k, 0, &mut vector::empty(), &mut result);
    };

    result
}

fun generate_combinations_recursive(
    variables: &vector<Point>,
    k: u64,
    start_idx: u64,
    current: &mut vector<Point>,
    result: &mut vector<vector<Point>>,
) {
    if (current.length() == k) {
        result.push_back(*current);
        return
    };

    let remaining = k - current.length();
    let available = variables.length() - start_idx;

    if (remaining > available) return;

    let mut i = start_idx;
    while (i < variables.length()) {
        current.push_back(variables[i]);
        generate_combinations_recursive(variables, k, i + 1, current, result);
        current.pop_back();
        i = i + 1;
    }
}

// --- DPLL Algorithm ---

/// Solve SAT instance using DPLL algorithm
#[allow(lint(public_random))]
public fun solve_sat(instance: &SATInstance, rng: &mut RandomGenerator): SATResult {
    let mut assignment = vec_map::empty<Point, bool>();
    let mut unassigned = instance.variables;

    // Add reasonable depth limit to prevent infinite recursion
    let max_depth = instance.variables.length() * 2;
    if (dpll_solve(&instance.clauses, &mut assignment, &mut unassigned, rng, max_depth)) {
        SATResult::Satisfiable(assignment)
    } else {
        SATResult::Unsatisfiable
    }
}

/// Core DPLL recursive algorithm
fun dpll_solve(
    clauses: &vector<Clause>,
    assignment: &mut VecMap<Point, bool>,
    unassigned: &mut vector<Point>,
    rng: &mut RandomGenerator,
    max_depth: u64,
): bool {
    // Depth limit to prevent infinite recursion
    if (max_depth == 0) {
        return false
    };
    // Check if all clauses are satisfied
    if (clauses.all!(|clause| clause_is_satisfied(clause, assignment))) {
        return true
    };

    // Check for conflicts (empty clauses)
    if (clauses.any!(|clause| clause_is_falsified(clause, assignment))) {
        return false
    };

    // Unit propagation with limit to prevent infinite loops
    let mut unit_prop_rounds = 0;
    loop {
        unit_prop_rounds = unit_prop_rounds + 1;
        if (unit_prop_rounds > unassigned.length() + 10) {
            // Prevent infinite unit propagation loops
            break
        };

        let mut unit_literal_opt = find_unit_literal(clauses, assignment);
        if (unit_literal_opt.is_none()) {
            break
        } else {
            let literal = unit_literal_opt.extract();
            // Assign the unit literal
            assignment.insert(literal.variable, literal.positive);
            unassigned.remove_value(&literal.variable);

            // Check for immediate conflict
            if (clauses.any!(|clause| clause_is_falsified(clause, assignment))) {
                return false
            }
        }
    };

    // Pure literal elimination
    let pure_literals = find_pure_literals(clauses, assignment, unassigned);
    pure_literals.do!(|literal| {
        assignment.insert(literal.variable, literal.positive);
        unassigned.remove_value(&literal.variable);
    });

    // Check if solved
    if (unassigned.is_empty()) {
        return clauses.all!(|clause| clause_is_satisfied(clause, assignment))
    };

    // Early termination: if we have too many variables left relative to depth, give up
    if (unassigned.length() > max_depth) {
        return false
    };

    // Choose branching variable (random selection)
    let r_u32 = rng.generate_u32();
    let branch_idx = (r_u32 as u64) % unassigned.length();
    let branch_var = unassigned[branch_idx];

    // Try both values (randomize order)
    let r_bool = rng.generate_bool();
    let values = if (r_bool) {
        vector[true, false]
    } else {
        vector[false, true]
    };

    values.any!(|value| {
        // Make assignment
        assignment.insert(branch_var, *value);
        unassigned.remove_value(&branch_var);

        // Recursive call with decremented depth
        let result = dpll_solve(clauses, assignment, unassigned, rng, max_depth - 1);

        // Backtrack if unsuccessful
        if (!result) {
            assignment.remove(&branch_var);
            unassigned.push_back(branch_var);
        };

        result
    })
}

/// Find a unit clause (clause with exactly one unassigned literal)
fun find_unit_literal(clauses: &vector<Clause>, assignment: &VecMap<Point, bool>): Option<Literal> {
    let mut i = 0;
    let mut res = option::none();
    while (i < clauses.length()) {
        let clause = clauses.borrow(i);
        i = i + 1;

        // Skip already satisfied clauses
        if (clause_is_satisfied(clause, assignment)) {
            continue
        };

        let unassigned_literals = clause_get_unassigned_literals(clause, assignment);

        if (unassigned_literals.length() == 1) {
            res = option::some(unassigned_literals[0]);
        };
    };

    res
}

/// Find pure literals (variables that appear with only one polarity)
fun find_pure_literals(
    clauses: &vector<Clause>,
    assignment: &VecMap<Point, bool>,
    unassigned: &vector<Point>,
): vector<Literal> {
    let mut pure_literals = vector::empty();

    unassigned.do_ref!(|var| {
        let mut appears_positive = false;
        let mut appears_negative = false;

        clauses.do_ref!(|clause| {
            // Skip satisfied clauses
            if (clause_is_satisfied(clause, assignment)) return;

            clause.literals.do_ref!(|lit| {
                if (lit.variable == *var) {
                    if (lit.positive) {
                        appears_positive = true;
                    } else {
                        appears_negative = true;
                    }
                }
            });
        });

        // If variable appears with only one polarity, it's pure
        if (appears_positive && !appears_negative) {
            pure_literals.push_back(Literal { variable: *var, positive: true });
        } else if (appears_negative && !appears_positive) {
            pure_literals.push_back(Literal { variable: *var, positive: false });
        }
    });

    pure_literals
}

// --- Deduction Analysis ---

/// Analyze SAT instance with proper stateless sampling and global consistency verification
#[allow(lint(public_random))]
public fun analyze_with_deductions(
    instance: &SATInstance,
    rng: &mut RandomGenerator,
): AnalysisResult {
    // First, get a sample solution
    let sample_result = solve_sat(instance, rng);

    let (sample_assignment, is_valid) = match (sample_result) {
        SATResult::Satisfiable(assignment) => {
            // Verify the sample satisfies ALL constraints globally
            let valid = verify_assignment_satisfies_all_constraints(instance, &assignment);
            (assignment, valid)
        },
        SATResult::Unsatisfiable => {
            // Return empty result for unsatisfiable instances
            return AnalysisResult {
                deductions: vec_map::empty(),
                sample_assignment: vec_map::empty(),
                is_valid: false,
            }
        },
    };

    // If sample doesn't satisfy all constraints, try multiple times for a valid sample
    let (final_sample, final_valid) = if (!is_valid) {
        find_valid_sample(instance, rng, 10) // Try up to 10 times
    } else {
        (sample_assignment, is_valid)
    };

    // Now test each variable for forced assignments using fresh instances
    let mut deductions = vec_map::empty<Point, DeducedState>();

    instance.variables.do_ref!(|var| {
        let deduced_state = test_variable_deduction_stateless(instance, *var, rng);
        deductions.insert(*var, deduced_state);
    });

    AnalysisResult {
        deductions,
        sample_assignment: final_sample,
        is_valid: final_valid,
    }
}

/// Test if a variable is forced to a particular value using stateless approach
fun test_variable_deduction_stateless(
    instance: &SATInstance,
    var: Point,
    rng: &mut RandomGenerator,
): DeducedState {
    // Test if variable can be true - create fresh instance
    let mut true_instance = create_fresh_sat_instance(&instance.variables);
    copy_constraints_to_instance(&instance.clauses, &mut true_instance);
    true_instance
        .clauses
        .push_back(Clause {
            literals: vector[Literal { variable: var, positive: true }],
        });

    let true_result = solve_sat(&true_instance, rng);
    let can_be_true = match (true_result) {
        SATResult::Satisfiable(assignment) => {
            // Verify this assignment satisfies all original constraints
            verify_assignment_satisfies_all_constraints(instance, &assignment)
        },
        _ => false,
    };

    // Test if variable can be false - create fresh instance
    let mut false_instance = create_fresh_sat_instance(&instance.variables);
    copy_constraints_to_instance(&instance.clauses, &mut false_instance);
    false_instance
        .clauses
        .push_back(Clause {
            literals: vector[Literal { variable: var, positive: false }],
        });

    let false_result = solve_sat(&false_instance, rng);
    let can_be_false = match (false_result) {
        SATResult::Satisfiable(assignment) => {
            // Verify this assignment satisfies all original constraints
            verify_assignment_satisfies_all_constraints(instance, &assignment)
        },
        _ => false,
    };

    if (can_be_true && can_be_false) {
        DeducedState::Undetermined
    } else if (can_be_true && !can_be_false) {
        DeducedState::ForcedTrue
    } else {
        DeducedState::ForcedFalse
    }
}

public fun is_satisfiable(state: &SATResult): bool {
    match (state) {
        SATResult::Unsatisfiable => false,
        _ => true,
    }
}

public fun deduce_bool(state: &DeducedState): Option<bool> {
    match (state) {
        DeducedState::ForcedTrue => option::some(true), // Must be mine
        DeducedState::ForcedFalse => option::some(false), // Must be safe
        DeducedState::Undetermined => option::none(), // Continue to random sampling
    }
}

/// Get a stateless, globally consistent sample for a single cell
/// This is the core function that ensures SPEC.md compliance
#[allow(lint(public_random))]
public fun sample_single_cell_consistent(
    variables: &vector<Point>,
    constraints: &vector<Clause>,
    target_cell: &Point,
    rng: &mut RandomGenerator,
): (bool, bool) {
    // Create fresh instance every time (stateless)
    let mut fresh_instance = create_fresh_sat_instance(variables);
    copy_constraints_to_instance(constraints, &mut fresh_instance);

    // Get multiple valid samples and verify consistency
    let (sample, is_valid) = find_valid_sample(&fresh_instance, rng, 10);

    if (!is_valid || !sample.contains(target_cell)) {
        return (false, false) // Default to safe, mark as invalid
    };

    let cell_state = *sample.get(target_cell);

    // Double-check that this assignment satisfies ALL constraints
    let satisfies_all = verify_assignment_satisfies_all_constraints(&fresh_instance, &sample);

    (cell_state, satisfies_all)
}

/// Create a fresh SAT instance with given variables (stateless)
fun create_fresh_sat_instance(variables: &vector<Point>): SATInstance {
    SATInstance {
        variables: *variables,
        clauses: vector::empty(),
    }
}

/// Copy constraints from one instance to another (for stateless operations)
fun copy_constraints_to_instance(source_clauses: &vector<Clause>, target: &mut SATInstance) {
    source_clauses.do_ref!(|clause| {
        target.clauses.push_back(*clause);
    });
}

/// Verify that an assignment satisfies ALL constraints in the instance
fun verify_assignment_satisfies_all_constraints(
    instance: &SATInstance,
    assignment: &VecMap<Point, bool>,
): bool {
    instance.clauses.all!(|clause| clause_is_satisfied(clause, assignment))
}

/// Attempt to find a valid sample that satisfies all constraints
fun find_valid_sample(
    instance: &SATInstance,
    rng: &mut RandomGenerator,
    max_attempts: u64,
): (VecMap<Point, bool>, bool) {
    let mut attempts = 0;

    while (attempts < max_attempts) {
        let sample_result = solve_sat(instance, rng);

        match (sample_result) {
            SATResult::Satisfiable(assignment) => {
                if (verify_assignment_satisfies_all_constraints(instance, &assignment)) {
                    return (assignment, true)
                }
            },
            SATResult::Unsatisfiable => {
                return (vec_map::empty(), false)
            },
        };

        attempts = attempts + 1;
    };

    // If we couldn't find a valid sample, return empty assignment
    (vec_map::empty(), false)
}

// --- Utility Functions ---

/// Stateless sampling function that creates fresh constraints and samples once
#[allow(lint(public_random))]
public fun sample_cell_stateless(
    variables: &vector<Point>,
    constraints: &vector<Clause>,
    target_cell: &Point,
    rng: &mut RandomGenerator,
): bool {
    // Create completely fresh SAT instance
    let mut fresh_instance = create_fresh_sat_instance(variables);
    copy_constraints_to_instance(constraints, &mut fresh_instance);

    // Get a valid sample
    let (sample, is_valid) = find_valid_sample(&fresh_instance, rng, 5);

    if (!is_valid) {
        // Fallback to safe if no valid sample found
        return false
    };

    // Return the state of the target cell
    if (sample.contains(target_cell)) {
        *sample.get(target_cell)
    } else {
        false // Default to safe
    }
}

/// Remove first occurrence of value from vector
use fun remove_value as vector.remove_value;

fun remove_value<T: copy + drop>(vec: &mut vector<T>, value: &T): bool {
    let mut i = 0;
    while (i < vec.length()) {
        if (&vec[i] == value) {
            vec.remove(i);
            return true
        };
        i = i + 1;
    };
    false
}

#[test_only]
/// Helper for testing - create simple SAT instance
public fun test_create_simple_instance(): (SATInstance, Point, Point) {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);

    let mut instance = sat_instance_new(vector[p1, p2]);

    // Add constraint: exactly 1 out of 2 variables
    sat_instance_add_exactly_k(&mut instance, &vector[p1, p2], 1);

    (instance, p1, p2)
}

#[test]
fun test_simple_sat_solving() {
    let (instance, p1, p2) = test_create_simple_instance();
    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");

    let result = solve_sat(&instance, &mut rng);

    // Should be satisfiable
    match (result) {
        SATResult::Satisfiable(assignment) => {
            // Exactly one should be true
            let p1_val = *assignment.get(&p1);
            let p2_val = *assignment.get(&p2);
            assert!(p1_val != p2_val, 0); // XOR: exactly one true
        },
        SATResult::Unsatisfiable => {
            assert!(false, 1); // Should not be unsatisfiable
        },
    }
}

#[test]
fun test_deduction_analysis() {
    let (instance, p1, p2) = test_create_simple_instance();
    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");

    let analysis = analyze_with_deductions(&instance, &mut rng);

    // Analysis should be valid
    assert!(analysis.is_valid, 0);

    // Both variables should be undetermined in symmetric case
    let p1_deduction = *analysis.deductions.get(&p1);
    let p2_deduction = *analysis.deductions.get(&p2);

    assert!(p1_deduction == DeducedState::Undetermined, 1);
    assert!(p2_deduction == DeducedState::Undetermined, 2);

    // Sample assignment should satisfy constraint (exactly one true)
    let p1_sample = *analysis.sample_assignment.get(&p1);
    let p2_sample = *analysis.sample_assignment.get(&p2);
    assert!(p1_sample != p2_sample, 3);

    // Verify the sample satisfies all constraints
    assert!(verify_assignment_satisfies_all_constraints(&instance, &analysis.sample_assignment), 4);
}

#[test]
fun test_sat_instance_creation() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let mut instance = sat_instance_new(vector[p1, p2]);

    let lit = literal_new(p1, true);
    let clause = clause_new(vector[lit]);
    sat_instance_add_clause(&mut instance, clause);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test");
    let result = solve_sat(&instance, &mut rng);

    match (result) {
        SATResult::Satisfiable(assignment) => {
            assert!(*assignment.get(&p1) == true, 0);
        },
        SATResult::Unsatisfiable => assert!(false, 1),
    }
}

#[test]
fun test_exactly_k_constraint_k0() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let mut instance = sat_instance_new(vector[p1, p2]);

    // Exactly 0 out of 2 variables
    sat_instance_add_exactly_k(&mut instance, &vector[p1, p2], 0);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test");
    let result = solve_sat(&instance, &mut rng);

    match (result) {
        SATResult::Satisfiable(assignment) => {
            assert!(*assignment.get(&p1) == false, 0);
            assert!(*assignment.get(&p2) == false, 1);
            // Verify global consistency
            assert!(verify_assignment_satisfies_all_constraints(&instance, &assignment), 2);
        },
        SATResult::Unsatisfiable => assert!(false, 3),
    }
}

#[test]
fun test_stateless_sampling() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let p3 = point::new(1, 0);
    let variables = vector[p1, p2, p3];

    let mut instance = sat_instance_new(variables);
    // Exactly 1 out of 3 variables
    sat_instance_add_exactly_k(&mut instance, &variables, 1);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"stateless_test");

    // Get a single consistent sample that satisfies all constraints
    let analysis = analyze_with_deductions(&instance, &mut rng);
    assert!(analysis.is_valid, 0);

    // From this consistent sample, check individual cells
    let sample1 = analysis.sample_assignment_cell_state(&p1);
    let sample2 = analysis.sample_assignment_cell_state(&p2);
    let sample3 = analysis.sample_assignment_cell_state(&p3);

    let s1 = if (sample1) { 1 } else { 0 };
    let s2 = if (sample2) { 1 } else { 0 };
    let s3 = if (sample3) { 1 } else { 0 };

    // Exactly one should be true (due to exactly 1 constraint)
    let true_count = s1 + s2 + s3;
    assert!(true_count == 1, 1);
}

#[test]
fun test_global_consistency_verification() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let mut instance = sat_instance_new(vector[p1, p2]);

    // Add constraint: exactly 1 out of 2
    sat_instance_add_exactly_k(&mut instance, &vector[p1, p2], 1);

    // Create valid assignment
    let mut valid_assignment = vec_map::empty();
    valid_assignment.insert(p1, true);
    valid_assignment.insert(p2, false);

    // Create invalid assignment
    let mut invalid_assignment = vec_map::empty();
    invalid_assignment.insert(p1, true);
    invalid_assignment.insert(p2, true);

    assert!(verify_assignment_satisfies_all_constraints(&instance, &valid_assignment), 0);
    assert!(!verify_assignment_satisfies_all_constraints(&instance, &invalid_assignment), 1);
}

#[test]
fun test_exactly_k_constraint_all() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let mut instance = sat_instance_new(vector[p1, p2]);

    // Exactly 2 out of 2 variables (all true)
    sat_instance_add_exactly_k(&mut instance, &vector[p1, p2], 2);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test");
    let result = solve_sat(&instance, &mut rng);

    match (result) {
        SATResult::Satisfiable(assignment) => {
            assert!(*assignment.get(&p1) == true, 0);
            assert!(*assignment.get(&p2) == true, 1);
        },
        SATResult::Unsatisfiable => assert!(false, 2),
    }
}

#[test]
fun test_unsatisfiable_instance() {
    use grid::point;

    let p1 = point::new(0, 0);
    let mut instance = sat_instance_new(vector[p1]);

    // Add contradictory clauses: p1 and !p1
    let clause1 = clause_new(vector[literal_new(p1, true)]);
    let clause2 = clause_new(vector[literal_new(p1, false)]);
    sat_instance_add_clause(&mut instance, clause1);
    sat_instance_add_clause(&mut instance, clause2);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test");
    let result = solve_sat(&instance, &mut rng);

    match (result) {
        SATResult::Satisfiable(_) => assert!(false, 0),
        SATResult::Unsatisfiable => {}, // Expected
    }
}

#[test]
fun test_forced_true_deduction() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let mut instance = sat_instance_new(vector[p1, p2]);

    // Add constraint: p1 must be true
    let clause = clause_new(vector[literal_new(p1, true)]);
    sat_instance_add_clause(&mut instance, clause);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test");
    let analysis = analyze_with_deductions(&instance, &mut rng);

    let deductions = analysis.deductions();
    let p1_deduction = deductions.get(&p1);
    let p2_deduction = deductions.get(&p2);

    assert!(p1_deduction == DeducedState::ForcedTrue, 0);
    assert!(p2_deduction == DeducedState::Undetermined, 1);
}

#[test]
fun test_forced_false_deduction() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let mut instance = sat_instance_new(vector[p1, p2]);

    // Add constraint: p1 must be false (!p1)
    let clause = clause_new(vector[literal_new(p1, false)]);
    sat_instance_add_clause(&mut instance, clause);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test");
    let analysis = analyze_with_deductions(&instance, &mut rng);

    let deductions = analysis.deductions();
    let p1_deduction = deductions.get(&p1);
    let p2_deduction = deductions.get(&p2);

    assert!(p1_deduction == DeducedState::ForcedFalse, 0);
    assert!(p2_deduction == DeducedState::Undetermined, 1);
}

#[test]
fun test_three_variable_exactly_one() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let p3 = point::new(1, 0);
    let mut instance = sat_instance_new(vector[p1, p2, p3]);

    // Exactly 1 out of 3 variables
    sat_instance_add_exactly_k(&mut instance, &vector[p1, p2, p3], 1);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test");
    let result = solve_sat(&instance, &mut rng);

    match (result) {
        SATResult::Satisfiable(assignment) => {
            let count = {
                let mut c = 0;
                if (*assignment.get(&p1)) c = c + 1;
                if (*assignment.get(&p2)) c = c + 1;
                if (*assignment.get(&p3)) c = c + 1;
                c
            };
            assert!(count == 1, 0);
        },
        SATResult::Unsatisfiable => assert!(false, 1),
    }
}

#[test]
fun test_impossible_constraint() {
    use grid::point;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);
    let mut instance = sat_instance_new(vector[p1, p2]);

    // Exactly 3 out of 2 variables (impossible)
    sat_instance_add_exactly_k(&mut instance, &vector[p1, p2], 3);

    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test");
    let result = solve_sat(&instance, &mut rng);

    match (result) {
        SATResult::Satisfiable(_) => assert!(false, 0),
        SATResult::Unsatisfiable => {}, // Expected
    }
}
