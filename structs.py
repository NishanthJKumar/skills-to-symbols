"""Structs
"""
from collections import namedtuple
import z3

class Option(namedtuple("Option", ["name", "is_applicable", "get_action",
                                   "is_terminal", "mask"])):
    """An option with a mask.
    """
    __slots__ = ()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Proposition(namedtuple("Proposition", ["name", "classifier", "origin"])):
    """A propositional symbol with a grounding classifier
    """
    __slots__ = ()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    @property
    def origin_str(self):
        return ", ".join([o.name for o in self.origin])
    


class Operator(namedtuple("Operator", ["name", "option", "preconditions",
                                       "positive_effects", "negative_effects",
                                       "conditional_effects"])):
    """An operator with preconditions and effects
    """
    __slots__ = ()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name



class DecisionTreeClassifier:
    """TODO: inherit from sklearn?

    Initialize a decision tree, perhaps by hand.

    The tuples are nested:
        [class] is a DT
        (DT, ([feature], [op], [threshold]), DT) is a DT.
            where if feature [op] threshold, then the
            first DT is used, else the second.

    Parameters
    ----------
    tuples : tuple
        See above
    """
    def __init__(self, tuples, feature_names=None):
        self._tuples = tuples
        self._feature_names = feature_names

    def __str__(self):
        if self._tuples is not None:
            return self._create_str_from_tuples(self._tuples)
        raise NotImplementedError()

    def _create_str_from_tuples(self, tuples, depth=0):
        """
        """
        if not isinstance(tuples, tuple):
            return str(tuples)
        feature, op, val = tuples[1]
        if self._feature_names is not None:
            feature = self._feature_names[feature]
        true_str = self._create_str_from_tuples(tuples[2], depth=depth+1)
        false_str = self._create_str_from_tuples(tuples[0], depth=depth+1)
        sp = " " * (2*depth)
        return f"if {feature}{op}{val}:\n{sp}  {true_str}\n{sp}else:\n{sp}  {false_str}"

    def get_dnf_clauses(self, tuples=None):
        """
        """
        # Base case
        if tuples is None:
            tuples = self._tuples
        # Leaf
        if not isinstance(tuples, tuple):
            assert tuples == True or tuples == False
            if tuples == True:
                return [set()]
            else:
                return []
        # Feature node
        false_child, (feature, op, val), true_child = tuples
        assert op in ["<", ">="]
        anti_op = "<" if op == ">=" else ">="
        true_clauses = self.get_dnf_clauses(tuples=true_child)
        false_clauses = self.get_dnf_clauses(tuples=false_child)
        for clause in true_clauses:
            clause.add((feature, op, val))
        for clause in false_clauses:
            clause.add((feature, anti_op, val))
        return true_clauses + false_clauses

    def get_all_features(self, tuples=None):
        if tuples is None:
            tuples = self._tuples
        if not isinstance(tuples, tuple):
            return set()
        left = self.get_all_features(tuples[0])
        right = self.get_all_features(tuples[2])
        return {tuples[1][0]} | left | right

    def project(self, features):
        """
        """
        tuples = self._get_projected_tuples(features, self._tuples)
        return DecisionTreeClassifier(tuples, feature_names=self._feature_names)

    def _get_projected_tuples(self, features, tuples):
        # Leaf
        if not isinstance(tuples, tuple):
            return tuples
        # Feature node
        if tuples[1][0] in features:
            # Projecting this out
            return self._get_projected_tuples(features, tuples[2])
        # No change
        return (self._get_projected_tuples(features, tuples[0]),
                tuples[1],
                self._get_projected_tuples(features, tuples[2]))

    def predict(self, x, tuples=None, verbose=False):
        """
        """
        # Base case
        if tuples is None:
            tuples = self._tuples
        # Leaf
        if not isinstance(tuples, tuple):
            if verbose: print("Returning", tuples)
            return tuples
        # Feature node
        feature, op, val = tuples[1]
        if op == "<":
            result = (x[feature] < val)
        else:
            assert op == ">="
            result = (x[feature] >= val)
        if verbose:
            feature_name = feature
            if self._feature_names is not None:
                feature_name = self._feature_names[feature]
            print(f"Checking {feature_name}{op}{val}...result is {result} [={x[feature]}].")
        if result:
            child = tuples[2]
        else:
            child = tuples[0]
        return self.predict(x, tuples=child, verbose=verbose)

    def __eq__(self, other, verbose=False):
        return self.issubset(other, verbose=verbose) and \
               other.issubset(self, verbose=verbose)

    def issubset(self, other, verbose=False):
        return not (self & other.negate()).issat(verbose=verbose)

    def negate(self):
        return DecisionTreeClassifier(
            self._negate_tuples(self._tuples),
            feature_names=self._feature_names)

    def _negate_tuples(self, tuples):
        # Leaf
        if not isinstance(tuples, tuple):
            assert tuples == True or tuples == False
            return not tuples
        # Feature node
        return (self._negate_tuples(tuples[0]),
                tuples[1],
                self._negate_tuples(tuples[2]))

    def __and__(self, other):
        assert isinstance(other, DecisionTreeClassifier)
        new_tuples = self._replace_true_leaves(self._tuples, other._tuples)
        return DecisionTreeClassifier(new_tuples,
            feature_names=self._feature_names)

    def _replace_true_leaves(self, tuples, other):
        # Leaf
        if not isinstance(tuples, tuple):
            assert tuples == True or tuples == False
            if tuples:
                return other
            return tuples
        # Feature node
        return (self._replace_true_leaves(tuples[0], other),
                tuples[1],
                self._replace_true_leaves(tuples[2], other))

    def issat(self, verbose=False):
        clauses = self.get_dnf_clauses()
        # Empty means False
        if len(clauses) == 0:
            return False
        # Get all variables
        feature_to_var = {}
        for clause in clauses:
            for conj in clause:
                feature = conj[0]
                if feature not in feature_to_var:
                    v = z3.Real(f'v{feature}')
                    feature_to_var[feature] = v
        # Create solver and add constraints
        s = z3.Solver()
        formula_constraints = []
        for clause in clauses:
            clause_constraints = []
            for (feature, op, val) in clause:
                v = feature_to_var[feature]
                if op == "<":
                    constraint = v < val
                else:
                    assert op == ">="
                    constraint = v >= val
                clause_constraints.append(constraint)
            formula_constraints.append(z3.And(*clause_constraints))
        s.add(z3.Or(*formula_constraints))
        result = str(s.check())
        assert result in ["sat", "unsat"]
        if verbose:
            model = s.model()
            print("Result:", result)
            for feat, vr in sorted(feature_to_var.items()):
                print(self._feature_names[feat], ":", model[vr])
        return result == "sat"

