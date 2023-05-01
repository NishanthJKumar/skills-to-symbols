import itertools
import re
import time
import subprocess


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


def run_planner(domain_file, problem_file, options):
    cmd_str = f"$FF_PATH -o {domain_file} -f {problem_file}"
    start_time = time.time()
    output = subprocess.getoutput(cmd_str)
    if "goal can be simplified to FALSE" in output or "unsolvable" in output:
        raise Exception("Plan not found with FF! Error: {}".format(output))
    ff_plan = re.findall(r"\d+?: (.+)", output.lower())
    if not ff_plan:
        raise Exception("Plan not found with FF! Error: {}".format(output))
    if ff_plan[-1] == "reach-goal":
        ff_plan = ff_plan[:-1]
    # Parse the plan into options
    option_name_to_option = {o.name : o for o in options}
    option_plan = []
    for plan_step in ff_plan:
        step_name, _ = plan_step.rsplit('-', 1)
        if step_name not in option_name_to_option:
            import ipdb; ipdb.set_trace()
            raise Exception("Failed to parse plan step: {}".format(plan_step))
        option_plan.append(option_name_to_option[step_name])
    return option_plan
        

