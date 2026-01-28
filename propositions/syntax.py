# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/syntax.py

"""Syntactic handling of propositional formulas."""

from __future__ import annotations
from functools import lru_cache
from typing import Mapping, Optional, Set, Tuple, Union

from logic_utils import frozen, memoized_parameterless_method


@lru_cache(maxsize=100)  # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= 'p' and string[0] <= 'z' and \
        (len(string) == 1 or string[1:].isdecimal())


@lru_cache(maxsize=100)  # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant, ``False`` otherwise.
    """
    return string == 'T' or string == 'F'


@lru_cache(maxsize=100)  # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'


@lru_cache(maxsize=100)  # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string in {'&', '|',  '->', '+', '<->', '-&', '-|'}


@frozen
class Formula:
    """An immutable propositional formula in tree representation, composed from
    variable names, and operators applied to them.

    Attributes:
        root (`str`): the constant, variable name, or operator at the root of
            the formula tree.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
    """
    root: str
    first: Optional[Formula]
    second: Optional[Formula]

    def __init__(self, root: str, first: Optional[Formula] = None,
                 second: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root operands.

        Parameters:
            root: the root for the formula tree.
            first: the first operand for the root, if the root is a unary or
                binary operator.
            second: the second operand for the root, if the root is a binary
                operator.
        """
        if is_variable(root) or is_constant(root):
            assert first is None and second is None
            self.root = root
        elif is_unary(root):
            assert first is not None and second is None
            self.root, self.first = root, first
        else:
            assert is_binary(root)
            assert first is not None and second is not None
            self.root, self.first, self.second = root, first, second

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        if is_variable(self.root) or is_constant(self.root):
            return self.root
        if is_unary(self.root):
            return self.root + repr(self.first)
        return '(' + repr(self.first) + self.root + repr(self.second) + ')'

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @memoized_parameterless_method
    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        if is_variable(self.root):
            return {self.root}
        if is_constant(self.root):
            return set()
        if is_unary(self.root):
            return self.first.variables()
        return self.first.variables().union(self.second.variables())

    @memoized_parameterless_method
    def operators(self) -> Set[str]:
        """Finds all operators in the current formula.

        Returns:
            A set of all operators (including ``'T'`` and ``'F'``) used in the
            current formula.
        """
        if is_variable(self.root):
            return set()
        if is_constant(self.root):
            return {self.root}
        if is_unary(self.root):
            return {self.root}.union(self.first.operators())
        return {self.root}.union(self.first.operators()).union(
            self.second.operators())

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Union[Formula, None], str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a variable name (e.g.,
            ``'x12'``) or a unary operator followed by a variable name, then the
            parsed prefix will include that entire variable name (and not just a
            part of it, such as ``'x1'``). If no prefix of the given string is a
            valid standard string representation of a formula then returned pair
            should be of ``None`` and an error message, where the error message
            is a string with some human-readable content.
        """
        if string == '':
            return None, 'Empty string'

        first_char = string[0]
        if is_variable(first_char):
            index = 1
            while index < len(string) and string[index].isdigit():
                index += 1
            return Formula(string[:index]), string[index:]
        if is_constant(first_char):
            return Formula(first_char), string[1:]

        if is_unary(first_char):
            operand, remainder = Formula._parse_prefix(string[1:])
            if operand is None:
                return None, remainder
            return Formula(first_char, operand), remainder

        if first_char == '(':
            first, remainder = Formula._parse_prefix(string[1:])
            if first is None:
                return None, remainder

            binary_candidates = ['<->', '->', '-&', '-|', '&', '|', '+']
            operator = None
            for candidate in binary_candidates:
                if remainder.startswith(candidate) and is_binary(candidate):
                    operator = candidate
                    remainder = remainder[len(candidate):]
                    break
            if operator is None:
                return None, 'Missing binary operator'
            second, remainder = Formula._parse_prefix(remainder)
            if second is None:
                return None, remainder
            if remainder == '':
                return None, 'Missing closing parenthesis'
            if remainder[0] != ')':
                return None, 'Missing closing parenthesis'
            return Formula(operator, first, second), remainder[1:]

        return None, 'Invalid prefix'

    @staticmethod
    def is_formula(string: str) -> bool:
        """Checks if the given string is a valid representation of a formula.

        Parameters:
            string: string to check.

        Returns:
            ``True`` if the given string is a valid standard string
            representation of a formula, ``False`` otherwise.
        """
        parsed, remainder = Formula._parse_prefix(string)
        return parsed is not None and remainder == ''

    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        assert Formula.is_formula(string)
        parsed, _ = Formula._parse_prefix(string)
        assert parsed is not None
        return parsed

    def polish(self) -> str:
        """Computes the polish notation representation of the current formula.

        Returns:
            The polish notation representation of the current formula.
        """
        if is_variable(self.root) or is_constant(self.root):
            return self.root
        if is_unary(self.root):
            return self.root + self.first.polish()
        return self.root + self.first.polish() + self.second.polish()

    @staticmethod
    def parse_polish(string: str) -> Formula:
        """Parses the given polish notation representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose polish notation representation is the given string.
        """
        def _parse_polish_prefix(start: int) -> Tuple[Union[Formula, None], int]:
            if start >= len(string):
                return None, start
            if string.startswith('~', start):
                operand, index = _parse_polish_prefix(start + 1)
                if operand is None:
                    return None, index
                return Formula('~', operand), index
            for operator in ['<->', '->', '-&', '-|', '&', '|', '+']:
                if string.startswith(operator, start):
                    first, index = _parse_polish_prefix(start + len(operator))
                    if first is None:
                        return None, index
                    second, index = _parse_polish_prefix(index)
                    if second is None:
                        return None, index
                    return Formula(operator, first, second), index
            ch = string[start]
            if is_constant(ch):
                return Formula(ch), start + 1
            if is_variable(ch):
                index = start + 1
                while index < len(string) and string[index].isdigit():
                    index += 1
                return Formula(string[start:index]), index
            return None, start

        parsed, index = _parse_polish_prefix(0)
        assert parsed is not None and index == len(string)
        return parsed

    def substitute_variables(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each variable name `v` that is a
        key in `substitution_map` with the formula `substitution_map[v]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            variable name occurrences originating in the current formula are
            substituted (i.e., variable name occurrences originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((p->p)|r)').substitute_variables(
            ...     {'p': Formula.parse('(q&r)'), 'r': Formula.parse('p')})
            (((q&r)->(q&r))|p)
        """
        for variable in substitution_map:
            assert is_variable(variable) 
        if is_variable(self.root):
            return substitution_map.get(self.root, self)
        if is_constant(self.root):
            return Formula(self.root)
        if is_unary(self.root):
            return Formula(self.root,
                           self.first.substitute_variables(substitution_map))
        return Formula(self.root,
                       self.first.substitute_variables(substitution_map),
                       self.second.substitute_variables(substitution_map))

    def substitute_operators(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each constant or operator `op`
        that is a key in `substitution_map` with the formula
        `substitution_map[op]` applied to its (zero or one or two) operands,
        where the first operand is used for every occurrence of ``'p'`` in the
        formula and the second for every occurrence of ``'q'``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            operator occurrences originating in the current formula are
            substituted (i.e., operator occurrences originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((x&y)&~z)').substitute_operators(
            ...     {'&': Formula.parse('~(~p|~q)')})
            ~(~~(~x|~y)|~~z)
        """
        for operator in substitution_map:
            assert is_constant(operator) or is_unary(operator) or \
                is_binary(operator)
            assert substitution_map[operator].variables().issubset({'p', 'q'})
        if is_variable(self.root):
            return Formula(self.root)
        if is_constant(self.root):
            if self.root in substitution_map:
                return substitution_map[self.root]
            return Formula(self.root)
        if is_unary(self.root):
            substituted_first = self.first.substitute_operators(
                substitution_map)
            if self.root in substitution_map:
                return substitution_map[self.root].substitute_variables(
                    {'p': substituted_first})
            return Formula(self.root, substituted_first)
        substituted_first = self.first.substitute_operators(substitution_map)
        substituted_second = self.second.substitute_operators(substitution_map)
        if self.root in substitution_map:
            return substitution_map[self.root].substitute_variables(
                {'p': substituted_first, 'q': substituted_second})
        return Formula(self.root, substituted_first, substituted_second)
