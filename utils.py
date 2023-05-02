import itertools
import re
import time
import subprocess
import sys
import os
import tempfile


def get_all_subsets(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def intersect_classifiers(classifiers):
    """Compute the intersection of classifiers

    Parameters
    ----------
    classifiers : [DecisionTreeClassifier]

    Returns
    -------
    intersection : DecisionTreeClassifier
    """
    if len(classifiers) == 1:
        return classifiers[0]
    intersection = classifiers[0]
    for other in classifiers[1:]:
        intersection &= other
    return intersection

def create_domain_file(operators, propositions, domain_name, filename):
    # Create predicates str (all zero-arity)
    predicates_str = "\n\t\t".join([f"({prop.name})" for prop in propositions])

    # Create operators str
    operators_str = ""
    for operator in operators:
        if len(operator.conditional_effects) > 0:
            raise NotImplementedError("TODO")

        preconditions_str = ""
        for prop in operator.preconditions:
            preconditions_str += f"\n\t\t\t({prop.name}) ; {prop.origin_str}"

        effects_str = ""
        for prop in operator.positive_effects:
            effects_str += f"\n\t\t\t({prop.name}) ; {prop.origin_str}"

        for prop in operator.negative_effects:
            effects_str += f"\n\t\t\t(not ({prop.name})) ; {prop.origin_str}"

        operators_str += f"""
    (:action {operator.name}
        :parameters ()
        :precondition (and {preconditions_str}
        )
        :effect (and {effects_str}
        )
    )
"""

    domain_str = f"""
(define (domain {domain_name})
    (:requirements :strips)
    (:predicates 
        {predicates_str}
    )
    {operators_str}
)
"""

    with open(filename, "w") as f:
        f.write(domain_str)

    print(f"Wrote out to {filename}.")

def create_problem_file(init, goal, domain_name, filename, problem_name=None):
    if not problem_name:
        problem_name = domain_name
    
    init_str = ""
    for prop in init:
        init_str += f"\n\t\t\t({prop.name})"

    goal_str = ""
    for prop in goal:
        goal_str += f"\n\t\t\t({prop.name})"

    problem_str = f"""
(define (problem {problem_name})
    (:domain {domain_name})
    (:objects )
    (:init {init_str})
    (:goal (and {goal_str}))
)
"""

    with open(filename, "w") as f:
        f.write(problem_str)

    print(f"Wrote out to {filename}.")


def run_planner(domain_file, problem_file, options, optimal):
    """A version of SeSamE that runs the Fast Downward planner to produce a
    single skeleton, then calls run_low_level_search() to turn it into a plan.

    Usage: Build and compile the Fast Downward planner, then set the environment
    variable FD_EXEC_PATH to point to the `downward` directory. For example:
    1) git clone https://github.com/aibasel/downward.git
    2) cd downward && ./build.py
    3) export FD_EXEC_PATH="<your path here>/downward"

    On MacOS, to use gtimeout:
    4) brew install coreutils

    Important Note: Fast Downward will potentially not work with null operators
    (i.e. operators that have an empty effect set). This happens when
    Fast Downward grounds the operators, null operators get pruned because they
    cannot help satisfy the goal. In A* search Discovered Failures could
    potentially add effects to null operators, but this ability is not
    implemented here.
    """
    timeout_cmd = "gtimeout" if sys.platform == "darwin" else "timeout"
    if optimal:
        alias_flag = "--alias seq-opt-lmcut"
    else:  # satisficing
        alias_flag = "--alias lama-first"
    # Run Fast Downward followed by cleanup. Capture the output.
    assert "FD_EXEC_PATH" in os.environ, \
        "Please follow the instructions in the docstring of this method!"
    fd_exec_path = os.environ["FD_EXEC_PATH"]
    exec_str = os.path.join(fd_exec_path, "fast-downward.py")
    # The SAS file is used when augmenting the grounded operators,
    # during dicovered failures, and it's important that we give
    # it a name, because otherwise Fast Downward uses a fixed
    # default name, which will cause issues if you run multiple
    # processes simultaneously.
    sas_file = tempfile.NamedTemporaryFile(delete=False).name
    # Run to generate sas
    cmd_str = (f"{timeout_cmd} {10} {exec_str} {alias_flag} "
               f"--sas-file {sas_file} {domain_file} {problem_file}")
    subprocess.getoutput(cmd_str)
    cmd_str = (f"{timeout_cmd} {10} {exec_str} {alias_flag} {sas_file}")
    output = subprocess.getoutput(cmd_str)
    cleanup_cmd_str = f"{exec_str} --cleanup"
    subprocess.getoutput(cleanup_cmd_str)
    # Extract the skeleton from the output and compute the atoms_sequence.
    if "Solution found!" not in output:
        raise ValueError(f"Plan not found with FD! Error: {output}")
    skeleton_str = re.findall(r"(.+) \(\d+?\)", output)
    if not skeleton_str:
        raise ValueError(f"Plan not found with FD! Error: {output}")
    # Parse the plan into options
    option_name_to_option = {o.name : o for o in options}
    option_plan = []
    for plan_step in skeleton_str:
        step_name, _ = plan_step.rsplit('-', 1)
        if step_name not in option_name_to_option:
            import ipdb; ipdb.set_trace()
            raise Exception("Failed to parse plan step: {}".format(plan_step))
        option_plan.append(option_name_to_option[step_name])
    return option_plan
        

