"""The main algorithms for constructing PDDL operators from options
"""
from structs import Option, Proposition, Operator
from utils import get_all_subsets, intersect_classifiers
import numpy as np
import itertools


def get_options_modifying_factor(factor, options):
    """Get the options modifying the variables in the factor.

    Denoted options(factor) in the paper.

    Parameters
    ----------
    factor : frozenset(int)
    options : {Option}

    Returns
    -------
    options : {Option}
    """
    return {o for i in factor for o in \
            get_options_modifying_state_var(i, options)}

def get_factors_for_option(option, factors):
    """Get the factors that an option modifies

    Denoted factors(option) in the paper.

    Parameters
    ----------
    option : Option
    factors : [frozenset(int)]

    Returns
    -------
    factors : [frozenset(int)]
    """
    return [frozenset(f) for f in factors \
            if get_options_modifying_factor(f, {option})]

def get_options_modifying_state_var(i, options):
    """Get the options modifying the state variable.

    Denoted modifies(s_i) in the paper.

    Parameters
    ----------
    i : int
        A state variable index
    options : {Option}

    Returns
    -------
    options : {Option}
    """
    return {o for o in options if o.mask[i]}

def get_factors_for_proposition(proposition, factors):
    """Get the factors that are referenced by the proposition's
    grounding classifier.

    Denoted factors(proposition) in the paper.

    Parameters
    ----------
    proposition : Proposition
    factors : [frozenset(int)]

    Returns
    -------
    factors : [frozenset(int)]
    """
    return get_factors_for_classifier(proposition.classifier, factors)

def get_factors_for_classifier(classifier, factors):
    """Get the factors that are referenced by the classifier

    Parameters
    ----------
    classifier : DecisionTreeClassifier
    factors : [frozenset(int)]

    Returns
    -------
    factors : [frozenset(int)]
    """
    return [f for f in factors if f & classifier.get_all_features()]

def factor_is_independent(f_i, e):
    """Check whether the factor f_i is independent in the effects e.

    NOTE: this is not really tested because there are no dependent
    effect sets in playroom.

    A factor is independent for this option's effect set if
    project(e, f_i) & project(e, f - f_i) == e.
    Note that project(e, f - f_i) is the set of values
    for the state variables in factor f_i that are possible
    to arrive at after executing the option. If the particular
    value depends on some other state variables, then the
    factor is not independent. This would manifest in the
    equation above because the intersection would include
    some states that are not in e. (The intersection should
    always be a superset of e, I think.)

    The paper does not specify how to perform this computation,
    but I think the following works for decision trees:
        - Find all constraints in e on any of the state variables
          in f_i.
        - For each branch in the decision tree (clause),
          determine whether all of the constraints hold. If not,
          conclude dependence.

    TODO: rewrite with SMT solver (z3)

    Parameters
    ----------
    f_i : frozenset(int)
        The factor
    e : DecisionTreeClassifier
        The effects

    Returns
    -------
    is_independent : bool
    """
    clauses = e.get_dnf_clauses()
    constraints = None
    for clause in clauses:
        # Perform projection by deleting irrelevant constraints
        projected_clause = {constraint for constraint in clause \
                            if constraint[0] in f_i}
        if constraints is None:
            constraints = projected_clause
        elif constraints != projected_clause:
            return False
    return True

def register_proposition(option, effects, proj_factors, propositions, symbol_count):
    """Create a propositional symbol corresponding to proj(effects, proj_factors)

    Parameters
    ----------
    option : Option
    effects : DecisionTreeClassifier
    proj_factors : [{int}]
    propositions : [Proposition]
    """
    proj_vars = [i for f in proj_factors for i in f]
    classifier = effects.project(proj_vars)
    # No use in adding always true or false proposition
    if str(classifier) in ["True", "False"]:
        return
    # Check if this proposition already exists
    for existing_prop in propositions:
        if str(existing_prop.classifier) == str(classifier):
            existing_prop.origin.append(option)
            return
    # Create new proposition
    name = f"symbol{next(symbol_count)}"
    proposition = Proposition(name, classifier, [option])
    # Add new proposition
    propositions.append(proposition)

def refers_to_effect(proposition, option):
    """Check whether the proposition classifies part of the
    option's effect.

    Parameters
    ----------
    proposition : Proposition
    option : option

    Returns
    -------
    refers : bool
    """
    return option in proposition.origin

def create_operators_from_options(options, option_to_preconditions,
                                  option_to_effects):
    """Generate a PDDL domain description from characterizing sets

    This is Algorithm 1 in Konidaris et al. (2018)
    http://cs.brown.edu/people/gdk/pubs/orig_sym_jair.pdf

    Parameters
    ----------
    options : {Option}
        A set of options.
    option_to_preconditions : {Option : DecisionTreeClassifier}
        Binary classifier describing the option's preconditions.
    option_to_effects : {Option : DecisionTreeClassifier}
        Binary classifier describing the option's effects.

    Returns
    -------
    operators : [Operator]
    propositions : [Proposition]
    """
    assert len(options) > 0, "Cannot create operators without options"
    assert set(options) == set(option_to_preconditions.keys())
    assert set(options) == set(option_to_effects.keys())
    symbol_count = itertools.count()

    # Extract dimensions etc. from the options
    state_dimension = len(next(iter(options)).mask) # called "n" in the paper
    assert all(len(o.mask) == state_dimension for o in options)

    # Initialize
    propositions, factors = [], []

    # Compute factors
    # Two state variables are in the same factor iff for any option,
    # the option either modifies both variables, or modifies neither.
    for i in range(state_dimension):
        # If there is some factor s.t. the options modifying the variables in
        # that factor == the options that change the value of s_i, then
        # we can add s_i to the factor.
        for factor in factors:
            if get_options_modifying_factor(factor, options) == \
               get_options_modifying_state_var(i, options):
                factor.add(i)
                break
        else:
            # Create a new factor containing just the new variable
            new_factor = {i}
            factors.append(new_factor)
    factors = [frozenset(f) for f in factors]

    # Generate symbol set
    for option in options:
        f = get_factors_for_option(option, factors)
        remaining_f = list(f)
        e = option_to_effects[option]

        # Identify independent factors
        for f_i in f:
            if factor_is_independent(f_i, e):
                # Add new symbol for this independent factor
                f_minus_i = [f_j for f_j in f if f_j != f_i]
                register_proposition(option, e, f_minus_i,
                                     propositions, symbol_count)
                # Remove from remaining factors
                remaining_f.remove(f_i)

        # Project out all combinations of remaining factors
        # WARNING: untested
        for f_s in get_all_subsets(remaining_f):
            if len(f_s) == 0:
                continue
            # Add new symbol
            register_proposition(option, e, f_s,
                                 propositions, symbol_count)
            propositions_for_option_effects[option].add(proposition)

    # Generate operator descriptions
    operators = []
    for option in options:
        pre = option_to_preconditions[option]
        factors_for_option = get_factors_for_option(option, factors)
        # Direct effects
        positive_effects = {prop for prop in propositions \
                            if refers_to_effect(prop, option)}
        other_props = [p for p in propositions if p not in positive_effects]
        # Negative effects
        negative_effects = []
        for prop in other_props:
            factors_for_prop = get_factors_for_proposition(prop, factors)
            # factors(proposition) subseteq factors(option)?
            if not set(factors_for_prop).issubset(set(factors_for_option)):
                continue
            # G(proposition) subseteq I(option)? # TODO NOTE THAT THIS IS WRONG IN THE PSEUDOCODE
            projected_pre = pre.project({v for f in set(factors) - set(factors_for_prop) for v in f})
            if not prop.classifier.issubset(projected_pre):
                continue
            negative_effects.append(prop)

        # Conditional effects
        # WARNING: untested
        conditional_effects = []
        for prop1 in other_props:
            # factors(prop1) \cap factors(option) != empty
            factors_for_prop1 = get_factors_for_proposition(prop1, factors)
            if not (set(factors_for_prop1) & set(factors_for_option)):
                continue
            # TODO: this is wrong in the same way that direct negative effects are wrong
            # G(prop1) subseteq pre
            if not prop1.classifier.issubset(pre):
                continue
            for prop2 in other_props:
                if prop1 is prop2:
                    continue
                # G(prop2) = Project(G(prop1), factors(option))
                if prop2.classifier == prop1.classifier.project(factors_for_option):
                    # Create conditional effect
                    conditional_effects.append((prop1, prop2))
                    break

        # Compute preconditions
        # Get the factors involved in the preconditions
        pre_factors = sorted(get_factors_for_classifier(pre, factors))
        # Enumerate all "assignments" of factors to symbols
        pre_factor_to_props = {f : set() for f in pre_factors}
        for prop in propositions:
            factors_for_prop = get_factors_for_proposition(prop, factors)
            for factor in set(factors_for_prop) & set(pre_factors):
                pre_factor_to_props[factor].add(prop)
        choices = [sorted(pre_factor_to_props[p]) for p in pre_factors]
        operator_name_count = itertools.count()
        visited_preconditions = set()
        for preconditions in itertools.product(*choices):
            preconditions = frozenset(preconditions)
            if preconditions in visited_preconditions:
                continue
            visited_preconditions.add(preconditions)
            # Empty if always True or False
            if len(preconditions) == 0:
                if str(pre) == "False":
                    continue
                assert str(pre) == "True"
            else:
                # Check whether these propositions are in the initiation set
                intersection = intersect_classifiers([p.classifier \
                                                      for p in preconditions])
                if not intersection.issubset(pre):
                    continue
            # We've found preconditions, create an operator
            operator_id = next(operator_name_count)
            name = f"{option.name}-{operator_id}"
            print(f"Creating operator {name}")
            operator = Operator(name, option, preconditions,
                                positive_effects, negative_effects,
                                conditional_effects)
            operators.append(operator)

    return operators, propositions


